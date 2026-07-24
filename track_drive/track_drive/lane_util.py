import numpy as np
import cv2

#BEV
# 출처: KUAC_2024-main lane_detection/src/utils.py warper() — 원본 픽셀좌표 그대로(640x150 ROI 기준, 반올림 없음)
#   src 원본: 좌하[0,126] 좌상[175,46] 우상[456,38] 우하[640,103]
#   dst 원본: x=169~489, y=0~150(ROI 높이 그대로)
#   같은 640x480 카메라·같은 ROI 높이(150px) 사용 가정 → 원본 수치/640,150 그대로 기입
#   (우리 점 순서인 좌상,우상,우하,좌하로 재배열만 함). 카메라 마운트 다르면 실차에서 재확인 필요.
BEV_SRC = np.float32([
        [175/640, 46/150],
        [456/640, 38/150],
        [640/640, 103/150],
        [0/640,   126/150],
    ])
BEV_DST = np.float32([
        [169/640, 0/150],
        [489/640, 0/150],
        [489/640, 150/150],
        [169/640, 150/150],
    ])


#Lane ROI
# 출처: KUAC_2024-main lane_detection/src/utils.py roi_for_lane() → image[246:396, :] (640x480 기준)
#   246/480=0.5125, 396/480=0.825 로 환산
LANE_ROI_TOP = 0.45
LANE_ROI_BOT = 0.825
#Debug
DEBUG_VIZ_LANE = True

# 흰 차선은 실차에서 전혀 검출되지 않아(실측 확인) 백색 검출 파이프라인을 걷어내고
# 노란색 중앙선 검출만으로 주행경로를 산출한다(차선 자체를 목표로 그대로 추종).

# BEV 유효영역 좌우 여유폭 — 커브 대응
#   BEV_DST가 정의하는 x범위(169~489/640, 폭 50%)만 유효영역으로 인정하면 커브
#   진입 시 노란선이 이 폭을 벗어나자마자 마스킹되어 사라져 직진으로 오인하는
#   문제가 있었다(실측 확인). BEV_DST 경계에서 좌우로 이 비율(roi_w 기준)만큼
#   더 여유를 줘서, 원근변환 사다리꼴 경계 밖(기하학적으로는 다소 부정확한
#   외삽 픽셀)이라도 색상 기반 검출은 계속 시도한다 — 좌표 정밀도보다 "커브에서
#   선을 놓치지 않는 것"이 우선이라는 판단. 실차 미검증 튜닝값.
BEV_VALID_MARGIN_RATIO = 0.15

# 구간별 무게중심(Moments) 기반 차선 추적
#   기존 슬라이딩 윈도우(14단 히스토그램 탐색 + 2차 polyfit + 이전 프레임 기반 탐색)를
#   걷어내고, ROI를 아래→위로 MOMENT_N_SLICES개 구간으로 나눠 구간마다 cv2.moments()로
#   무게중심(cx = M10/M00)만 구하는 방식으로 단순화한다. 구간마다 완전히 독립적으로
#   계산하므로 한 구간이 반사/잡음으로 오염돼도 다른 구간엔 전혀 영향이 없다.
#   점들은 직선이 아니라 다항회귀(_fit_curve, 아래 SlideWindow) 곡선으로 이어
#   커브 형태를 완만하게 근사한다.
#   실차 미검증 튜닝값.
MOMENT_N_SLICES = 8        # ROI를 세로로 나눌 구간 수(아래→위 — 많을수록 회귀 곡선이 촘촘해지지만 구간당 픽셀이 줄어 노이즈에 약해짐)
MOMENT_MIN_PIXELS = 15     # 구간 내 노랑 픽셀수(M00)가 이 미만이면 그 구간은 "차선 없음" 처리
MOMENT_NEAR_SLICES = 2     # near_center(조향용 근거리 편차) 평가에 쓸 아래쪽(근거리) 구간 수
MOMENT_FAR_SLICES = 2      # far_center(코너 예측용 원거리 편차) 평가에 쓸 위쪽(원거리) 구간 수

# 구간 간 일관성 체크 (반사광 등 이상치 슬라이스 제거)
#   문제 : 반사광은 특정 슬라이스 하나만 뜬금없이 튄 무게중심을 만들기 쉽다.
#         밝기/색상만으로는 진짜 차선과 반사광을 못 가른다는 게 이미 여러 번
#         확인됐으므로, 완전히 다른 축인 "슬라이스끼리의 위치 일관성"으로
#         걸러낸다 — 진짜 차선은 슬라이스가 위로 갈수록 완만하게 이어지지만
#         반사광은 그 흐름에서 혼자 튄다.
#   방법 : 이번 프레임의 유효 슬라이스들에 1차(직선) 최소자승 추세선을 맞추고,
#         그 추세에서 LANE_SLICE_OUTLIER_MAX(px) 이상 벗어난 슬라이스만 제외한다.
#         프레임 간 기억이 필요 없어 가볍고(1패스, 폴백 없음), ROI가 짧아서
#         커브 구간에서도 직선 근사가 크게 어긋나지 않는다.
#   실차 미검증 튜닝값.
LANE_SLICE_OUTLIER_MAX = 40   # 추세선에서 이 이상(px) 벗어난 슬라이스는 이상치로 제외
LANE_SLICE_FIT_MIN = 3        # 유효 슬라이스가 이 개수 미만이면 추세 판단이 불안정하므로 검사 자체를 생략


#def Debugging(flag):
# DEBUG 상수들 모아놓자(외부 파일로 뺄지는 고민해보기)
class CameraProcessor:
    def __init__(self):
        self.roi = None
        self.bev = None
        self.yellow = None

        self.roi_h = 0
        self.roi_w = 0

    def processor(self, frame):
        # ROI
        h, w = frame.shape[:2]

        self.roi = frame[
            int(h*LANE_ROI_TOP): int(h*LANE_ROI_BOT), :
            ]
        self.roi_h, self.roi_w = self.roi.shape[:2]

        #BEV
        bev_src = BEV_SRC*np.array(
            [self.roi_w, self.roi_h],
            dtype = np.float32
        )

        bev_dst = BEV_DST*np.array(
            [self.roi_w, self.roi_h],
            dtype = np.float32
        )

        M = cv2.getPerspectiveTransform(
            bev_src, bev_dst
        )
        self.bev = cv2.warpPerspective(
            self.roi,
            M,
            (self.roi_w, self.roi_h)
        )

        # HSV
        hsv = cv2.cvtColor(self.bev, cv2.COLOR_BGR2HSV)

        #Yellow Lane
        self.yellow = cv2.inRange(
            hsv, np.array([15,80,80]), np.array([40,255,255])
        )

        # BEV 워프 유효 영역 밖(원근변환 사다리꼴 바깥) 마스킹
        #   BEV_DST가 정의하는 x범위(169~489/640 비율) 밖은 실제 도로가 아니라
        #   getPerspectiveTransform이 원근평면을 무한히 확장해서 만들어낸 외삽
        #   픽셀이다. 다만 그 경계선을 곧이곧대로 유효영역으로 쓰면 커브 진입 시
        #   노란선이 사다리꼴 밖으로 살짝만 밀려도 바로 지워져 직진으로 오인한다
        #   (실측 확인). BEV_VALID_MARGIN_RATIO만큼 좌우로 여유를 더 줘서 다소
        #   부정확한 외삽 영역까지 색상 검출을 허용한다(좌표 정밀도보다 놓치지
        #   않는 게 우선).
        margin_px = BEV_VALID_MARGIN_RATIO * self.roi_w
        valid_x_lo = max(0, int(BEV_DST[0, 0] * self.roi_w - margin_px))
        valid_x_hi = min(self.roi_w, int(BEV_DST[1, 0] * self.roi_w + margin_px))
        self.yellow[:, :valid_x_lo] = 0
        self.yellow[:, valid_x_hi:] = 0

        #Morphology
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (3,3)
        )

        # yellow morphology
        self.yellow = cv2.morphologyEx(
            self.yellow, cv2.MORPH_CLOSE, kernel
        )

        # Connected Components
        num, labels, stats, _ = cv2.connectedComponentsWithStats(self.yellow)

        mask = np.zeros_like(self.yellow)

        for i in range(1, num):
            area = stats[i, cv2.CC_STAT_AREA]

            width = stats[i, cv2.CC_STAT_WIDTH]

            if area>20 and width < 80:
                mask[labels == i] = 255

        self.yellow = mask

        #DEBUG
        if DEBUG_VIZ_LANE:
            cv2.imshow("lane_bev", self.bev)
            cv2.imshow("lane_yellow", self.yellow)

        return (self.bev, self.yellow)

class SlideWindow:
    """
    이름은 track_drive.py와의 인터페이스 호환을 위해 SlideWindow로 유지하지만,
    내부 구현은 "히스토그램 슬라이딩 윈도우 + 2차 polyfit + 이전 프레임 기반 탐색"
    대신 "구간별 무게중심(Moments)" 방식으로 교체됐다. ROI를 위아래로 나눠
    구간마다 cv2.moments()로 무게중심만 구하고, 그 점들을 직선이 아니라
    다항회귀(_fit_curve) 곡선으로 이어 커브 형태를 완만하게 근사한다.
    한 구간이 반사/점선 틈으로 비어도 다른 구간들의 추세로 메꿔진다.

    흰 차선은 실차에서 검출이 안 되는 게 확인되어 제외하고, 노란 중앙선
    검출만으로 주행목표(회귀곡선의 근거리/원거리 지점)를 산출한다.
    """
    def __init__(self):
        self.vis = None

        self.roi_h = 0
        self.roi_w = 0

        # 구간별 무게중심 저장 — calc_center()/시각화에서 재사용
        #   리스트 길이는 MOMENT_N_SLICES, 인덱스 0=가장 아래(근거리) ~ 마지막=가장 위(원거리)
        self.yellow_centers = []

    def _slice_centers(self, mask, x_offset, color):
        """
        mask를 아래→위로 MOMENT_N_SLICES개 구간으로 나눠 구간마다 cv2.moments()로
        무게중심(cx)을 구한다. 슬라이딩 윈도우처럼 이전 구간 위치를 이어받지
        않고 구간마다 독립적으로 계산하므로, 한 구간이 반사/잡음으로 날아가도
        다른 구간에는 전혀 영향이 없다.
          입력 : mask     — 이진마스크(좌/우 절반 등으로 이미 열 방향이 잘려있을 수 있음)
                x_offset — mask가 원본 ROI에서 잘려나온 경우의 x좌표 보정값(좌우 분리용)
                color    — 디버그 사각형 색상
          출력 : centers — 길이 MOMENT_N_SLICES 리스트. 각 원소는 (y_center, cx) 또는
                 해당 구간 픽셀수가 MOMENT_MIN_PIXELS 미만이면 None
        """
        slice_h = self.roi_h // MOMENT_N_SLICES
        centers = []

        for i in range(MOMENT_N_SLICES):
            y_high = self.roi_h - i * slice_h
            # 마지막(가장 위) 구간은 정수나눗셈 나머지를 포함해 y=0까지 전부 커버
            y_low = 0 if i == MOMENT_N_SLICES - 1 else self.roi_h - (i + 1) * slice_h

            band = mask[y_low:y_high, :]

            if DEBUG_VIZ_LANE:
                cv2.rectangle(
                    self.vis,
                    (x_offset, y_low), (x_offset + mask.shape[1] - 1, max(y_high - 1, y_low)),
                    color, 1
                )

            M = cv2.moments(band, binaryImage=True)

            if M['m00'] < MOMENT_MIN_PIXELS:
                centers.append(None)
                continue

            cx = M['m10'] / M['m00'] + x_offset
            y_center = (y_low + y_high) / 2.0
            centers.append((y_center, cx))

        return centers

    def _reject_outliers(self, centers):
        """
        구간별 무게중심들 사이의 위치 일관성을 체크해서 이상치(반사광 등으로
        혼자 튄 값)를 제거한다. 진짜 차선은 슬라이스가 위로 갈수록 완만하게
        이어지지만, 반사광은 특정 슬라이스 하나만 뜬금없이 튀는 값을 만들기
        쉽다. 프레임 간 기억 없이 "이번 프레임 슬라이스들끼리"만 비교한다.
          방법 : Leave-one-out — 각 슬라이스를 검사할 때 "자기 자신을 뺀"
                나머지 유효 슬라이스들로만 1차(직선) 추세선을 맞추고, 그
                추세에서 LANE_SLICE_OUTLIER_MAX(px) 이상 벗어나면 이상치로
                None 처리한다. 전체 점으로 한 번에 추세선을 맞추면 이상치
                자신이 추세선을 자기 쪽으로 끌어당겨서 오히려 안 걸러지는
                문제가 있어(실측으로 확인됨) 반드시 leave-one-out으로 한다.
                슬라이스가 3~5개뿐이라 매 프레임 여러 번 다시 피팅해도 비용은
                무시할 수준이다.
          입력/출력 : centers — _slice_centers()와 동일한 형식(길이 불변)
        """
        valid_idx = [i for i, c in enumerate(centers) if c is not None]
        if len(valid_idx) < LANE_SLICE_FIT_MIN:
            return centers   # 점이 너무 적으면 추세 판단 자체가 불안정 → 검사 생략

        ys = np.array([centers[i][0] for i in valid_idx])
        xs = np.array([centers[i][1] for i in valid_idx])

        result = list(centers)

        for k, i in enumerate(valid_idx):
            other_y = np.delete(ys, k)
            other_x = np.delete(xs, k)

            if len(other_y) < 2:
                continue   # 나머지가 1개뿐이면 직선을 정의할 수 없어 검사 생략

            slope, intercept = np.polyfit(other_y, other_x, 1)
            y_i, x_i = centers[i]
            residual = abs(x_i - (slope * y_i + intercept))

            if residual > LANE_SLICE_OUTLIER_MAX:
                result[i] = None
                if DEBUG_VIZ_LANE:
                    cv2.drawMarker(
                        self.vis,
                        (int(np.clip(x_i, 0, self.roi_w - 1)), int(y_i)),
                        (0, 0, 255), cv2.MARKER_TILTED_CROSS, 12, 2
                    )

        return result

    def _band_y_centers(self):
        """
        _slice_centers()와 동일한 기하학으로 MOMENT_N_SLICES개 구간의 y중심 좌표만
        계산한다(픽셀 유무와 무관, 순수 기하학). near_y/far_y 평가지점을 구할 때
        해당 구간이 실제로 검출됐는지 여부와 상관없이 "그 위치의 곡선값"을 뽑기
        위해 쓴다.
        """
        slice_h = self.roi_h // MOMENT_N_SLICES
        ys = []
        for i in range(MOMENT_N_SLICES):
            y_high = self.roi_h - i * slice_h
            y_low = 0 if i == MOMENT_N_SLICES - 1 else self.roi_h - (i + 1) * slice_h
            ys.append((y_low + y_high) / 2.0)
        return ys

    def _fit_curve(self, centers):
        """
        유효한 (y, cx) 점들에 다항회귀를 맞춰 x = f(y) 곡선(계수)을 반환한다.
        점 사이를 직선으로 잇는 대신 완만한 곡선으로 커브 형태를 근사하기 위함
        —  slice 하나가 반사/점선 틈으로 비어도 나머지 점들의 추세로 메꿔진다.
        점 개수가 적을수록 과적합을 피하려고 차수를 낮춘다(3개 이상=2차,
        2개=1차, 1개=상수). 유효 점이 하나도 없으면 None.
        """
        pts = [c for c in centers if c is not None]
        if not pts:
            return None

        ys = np.array([p[0] for p in pts])
        xs = np.array([p[1] for p in pts])
        degree = 2 if len(pts) >= 3 else (1 if len(pts) >= 2 else 0)

        return np.polyfit(ys, xs, degree)

    def draw_centers(self, centers, color):
        """구간별 무게중심을 점으로만 찍는다(연결은 _draw_curve의 회귀곡선이 담당)."""
        for c in centers:
            if c is None:
                continue
            y, cx = c
            cv2.circle(self.vis, (int(np.clip(cx, 0, self.roi_w - 1)), int(y)), 4, color, -1)

    def _draw_curve(self, curve, color, samples=30):
        """다항회귀 곡선을 y방향으로 촘촘히 샘플링해 완만한 곡선으로 그린다(직선 연결 대체)."""
        ys = np.linspace(0, self.roi_h - 1, samples)
        xs = np.clip(np.polyval(curve, ys), 0, self.roi_w - 1)
        pts = np.stack([xs, ys], axis=1).astype(np.int32)
        cv2.polylines(self.vis, [pts], False, color, 2)

    def visualize(self, offset, curve):
        self.draw_centers(self.yellow_centers, (0,165,255))
        if curve is not None:
            self._draw_curve(curve, (255,0,255))

        cv2.line(
            self.vis, (self.roi_w//2, 0), (self.roi_w//2, self.roi_h),
            (0,0,255), 1
        )
        cv2.putText(
            self.vis,
            f'offset : {offset:+.1f}',
            (10,25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255,255,255),
            2
        )

        if DEBUG_VIZ_LANE:
            cv2.imshow("lane_result", self.vis)
            cv2.waitKey(1)

    def calc_center(self):
        curve = self._fit_curve(self.yellow_centers)

        lane_valid = curve is not None
        offset = 0
        lookahead = 0

        # 회귀곡선에서 근거리/원거리 대표 y지점의 x값을 평가해 offset/lookahead를 구한다.
        # 원본 슬라이스 평균(그 슬라이스가 검출됐을 때만 유효) 대신 곡선을 쓰면, 딱 그
        # 대표 y지점 슬라이스가 반사/점선 틈으로 비어 있어도 다른 슬라이스들의 추세로
        # 값을 뽑아낼 수 있어 커브 진입 구간에서 더 안정적이다.
        if lane_valid:
            band_ys = self._band_y_centers()
            near_y = float(np.mean(band_ys[:MOMENT_NEAR_SLICES]))
            far_y  = float(np.mean(band_ys[-MOMENT_FAR_SLICES:]))

            near_center = float(np.polyval(curve, near_y))
            far_center  = float(np.polyval(curve, far_y))

            offset = near_center - self.roi_w / 2
            lookahead = far_center - self.roi_w / 2

        lane_center = self.roi_w / 2 + offset

        self.visualize(offset, curve)

        return lane_valid, offset, lookahead, lane_center

    def detect(self, bev, yellow):
        self.roi_h, self.roi_w = yellow.shape
        self.vis = bev.copy()

        # BEV 유효영역 마스킹(CameraProcessor.processor)이 이미 열 방향을 걸러주므로
        # 여기서 별도로 폭을 다시 자르지 않는다(중복 제한 시 커브에서 선을 놓치기 쉬움).
        self.yellow_centers = self._reject_outliers(self._slice_centers(
            yellow, 0, (0,180,255)
        ))

        return self.calc_center()
