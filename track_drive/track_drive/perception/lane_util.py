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
# TOP을 더 낮은 비율로 내려서(화면 위쪽=먼 거리) ROI를 더 멀리까지 보도록 확장.
LANE_ROI_TOP = 0.35
LANE_ROI_BOT = 0.825
#Debug — config.py에서 가져온다(실차 테스트 중 값을 바꾸려면 config.py를 고칠 것)
from ..config import DEBUG_VIZ_LANE

# Adaptive Thresholding — 흰 차선 마스킹
#   기존 CLAHE+Top-Hat(21x21 국소대비)+고정 임계값(10)+HSV 조합 대신, CLAHE 다음
#   단계를 cv2.adaptiveThreshold 하나로 단순화한다. 픽셀마다 자기 주변 blockSize
#   이웃의 (가우시안 가중)평균 밝기를 기준으로 임계값을 다시 계산해주기 때문에,
#   트랙 전체 조명이 불균일해도(한쪽은 밝고 한쪽은 그늘) 별도 튜닝 없이 안정적으로
#   차선을 잡을 수 있다. THRESH_BINARY라 "주변 평균-C 보다 밝은 픽셀"이 흰색(255)이 된다.
#   ※ 주의: 정반사(글레어)도 "주변보다 국소적으로 밝다"는 같은 성질을 가지므로
#     adaptiveThreshold 하나로 반사광이 완전히 걸러지는 건 아니다 — 이후 CCA
#     면적 필터로 작은 반사 얼룩 위주로 추가 제거한다(아래 CameraProcessor.processor 참고).
#   ※★ C는 반드시 음수여야 한다(실측으로 확인됨, 아래 참고) ★
#     T(x,y) = 주변평균 - C 이므로 C>0이면 T가 주변평균보다 낮아져서, 텍스처가
#     거의 없는 평평한 바닥(에폭시처럼 매끈한 면)에서는 "픽셀값 ≈ 주변평균"이라
#     사실상 전 영역이 threshold를 통과해버린다(합성 테스트 실측: C=+2일 때
#     균일한 바닥의 96.6%가 흰색으로 오검출됨 — 차선 유무와 무관하게 화면 전체가
#     "차선"으로 잡히는 셈이라 반사광보다 더 위험한 실패모드).
#     C를 음수로 주면 T=주변평균+|C|가 되어 "주변보다 확실히 밝아야만" 통과하고,
#     평평한 바닥은 정상적으로 검게 남는다(동일 테스트에서 C=-10부터 오검출 0%,
#     동시에 실제 차선 검출률은 C=+2일 때와 동일하게 유지됨 — 손해 없이 안전해짐).
ADAPTIVE_BLOCK_SIZE = 41   # 로컬 평균을 낼 이웃 크기(px, 홀수 필수) — 클수록 더 넓은 영역의 평균과 비교
ADAPTIVE_C = -8        # 주변평균보다 이만큼(px 밝기값) 더 밝아야 흰색 인정. 반드시 음수로 유지할 것

# Median Blur — 폭(굵기) 기준으로 얇은 반사 줄무늬 제거
#   문제 : 골진(주름진) 반사면이 조명을 여러 갈래 가는 대각선 줄무늬로 반사시키는
#     경우, adaptiveThreshold(밝기 기준)나 구간 간 일관성 체크(위치 기준)로는
#     못 걸러진다 — 반사 줄무늬 하나하나가 그 자체로 밝고 매끄럽게 이어지는
#     "그럴듯한 선"이기 때문. 지금까지의 밝기/일관성 축과 다른 "폭" 축으로 접근한다.
#   해결 : adaptiveThreshold 전에 그레이스케일에 medianBlur를 적용. 커널 안에서
#     다수결로 값을 정하는 특성상, 커널 폭의 절반보다 얇은 밝은 줄무늬는 주변
#     어두운 배경에 묻혀 사라지고, 그보다 굵은 실제 차선은 살아남는다.
#   ※ 전제 조건: "반사 줄무늬 폭 < MEDIAN_BLUR_KSIZE/2 < 실제 차선 폭"이 실측으로
#     성립해야 한다. 두 폭이 비슷하면 반사도 안 지워지거나, 반대로 점선/커브 구간의
#     가늘어 보이는 실제 차선까지 같이 지워질 수 있다 — 실차 영상에서 lane_white
#     디버그 창으로 두 폭을 비교해보고 커널 크기를 재조정할 것.
#   실차 미검증 튜닝값.
MEDIAN_BLUR_KSIZE = 11 #13      # 홀수 필수. 이 값의 절반(≈4~5px)보다 얇은 밝은 줄무늬를 지운다

# Canny Edge Detection — adaptiveThreshold가 놓치는 저대비 차선 경계 보강
#   adaptiveThreshold는 "주변 평균보다 밝은가"라는 밝기 기준 하나만 보기 때문에,
#   그림자가 걸치거나 노면과 명암차가 작은 구간은 문턱을 못 넘어 통째로 비어버릴
#   수 있다. Canny는 밝기 절대값이 아니라 그레이디언트(변화량)를 보므로 밝기
#   기준과는 독립적인 실패모드를 갖는다 — 두 마스크를 합집합(OR)으로 합치면
#   한쪽이 놓친 차선 경계를 다른 쪽이 보완해준다.
#   GaussianBlur는 (이미 medianBlur를 거친 gray 위에 한 번 더) Canny의 그레이디언트
#   계산이 센서 노이즈에 과민 반응해 잔가지 엣지를 만드는 걸 막는 표준 전처리다.
#   실차 미검증 튜닝값.
GAUSSIAN_BLUR_KSIZE = 5    # 홀수 필수. Canny 전 노이즈 억제용
CANNY_LOW = 80    #100         # 약한 엣지를 강한 엣지에 연결할지 판단하는 하위 임계값
CANNY_HIGH = 200  #220        # 이 값을 넘으면 바로 강한 엣지로 확정(보통 LOW의 2~3배 권장)

# 구간별 무게중심(Moments) 기반 차선 추적
#   기존 슬라이딩 윈도우(14단 히스토그램 탐색 + 2차 polyfit + 이전 프레임 기반 탐색)를
#   걷어내고, ROI를 아래→위로 MOMENT_N_SLICES개 구간으로 나눠 구간마다 cv2.moments()로
#   무게중심(cx = M10/M00)만 구하는 방식으로 단순화한다. 구간마다 완전히 독립적으로
#   계산하므로 한 구간이 반사/잡음으로 오염돼도 다른 구간엔 전혀 영향이 없고,
#   폴리피팅 없이 구간별 점을 그대로 이으면 커브 형태를 직관적으로 따라간다.
#   실차 미검증 튜닝값.
MOMENT_N_SLICES = 5        # ROI를 세로로 나눌 구간 수(아래→위, 3~5 권장 — 많을수록 커브 추종이 촘촘해지지만 구간당 픽셀이 줄어 노이즈에 약해짐)
MOMENT_MIN_PIXELS = 15     # 구간 내 흰/노랑 픽셀수(M00)가 이 미만이면 그 구간은 "차선 없음" 처리
MOMENT_NEAR_SLICES = 2     # near_center(조향용 근거리 편차) 계산에 쓸 아래쪽(근거리) 구간 수
MOMENT_FAR_SLICES = 2      # far_center(코너 예측용 원거리 편차) 계산에 쓸 위쪽(원거리) 구간 수

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

# 흰색 마스크 Connected Components 최소 면적
#   기존 area>5는 1~2px짜리 점만 걸러져 실제 반사 얼룩(보통 수십 px)은 그대로
#   통과했다. MOMENT_MIN_PIXELS와 같은 기준으로 맞춰 실질적으로 걸러지게 한다.
WHITE_CCA_MIN_AREA = 15

# 프레임 간 스파이크 필터링(디바운스) — STABLE_FRAME_MIN/STABLE_JUMP_MAX는
# config.py에 있다(설계 배경 주석도 그쪽에). 반사/워프 왜곡 등으로 특정 프레임
# 하나만 offset이 크게 튀는 걸 걸러내는 용도 — 실차 테스트 중 값을 바꾸려면
# config.py를 고칠 것.
from ..config import STABLE_FRAME_MIN, STABLE_JUMP_MAX

# 명시적 경로(웨이포인트) 생성 — controller/pure_pursuit.py가 소비
#   기존 offset/lookahead(근거리·원거리 두 그룹의 평균 스칼라)만으로는 Pure Pursuit이
#   목표점을 고를 "경로"가 없다. 그래서 슬라이스별 중심점(구간마다 독립 계산된 값,
#   반사광 등은 이미 _reject_outliers()로 제외된 상태)을 선형보간(np.interp)으로 이어
#   ROI 픽셀 좌표계의 명시적 경로를 만들고, 그 위를 일정 간격으로 샘플링해 웨이포인트로
#   쓴다([2026-08-07] 원래 2차 다항식 피팅+외삽이었다 — _fit_and_sample_path() 주석 참고).
#   실차 미검증 튜닝값.
PATH_N_WAYPOINTS = 12       # 경로에서 뽑아낼 웨이포인트 수

# 경로(웨이포인트) 프레임 간 EMA 스무딩 — PATH_EMA_ALPHA는 config.py에 있다
# (설계 배경/변경 이력 주석도 그쪽에 있음). 실차 테스트 중 값을 바꾸려면
# config.py를 고칠 것.
from ..config import PATH_EMA_ALPHA


#def Debugging(flag):
# DEBUG 상수들 모아놓자(외부 파일로 뺄지는 고민해보기)
class CameraProcessor:
    def __init__(self):
        self.roi = None
        self.bev = None
        self.white = None
        self.yellow = None
        self.edges = None

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

        # BEV_SRC 사다리꼴 밖(도로가 아닌 배경) 마스킹 — 워프 전에 미리 제거
        #   warpPerspective는 ROI 전체를 투영하기 때문에, BEV_SRC가 실제 도로
        #   경계와 안 맞으면(카메라 마운트 차이 등, 이 값은 다른 팀 캘리브레이션을
        #   그대로 가져온 값이라 우리 카메라에서 재확인 안 됨) 도로 바깥 배경까지
        #   같이 심하게 늘려펴져 bev로 끌려온다(실측: 대각선으로 왜곡된 배경 줄무늬가
        #   Canny/adaptiveThreshold를 오검출시켜 lane_white에 섞여 들어감 — 아래
        #   valid_x 컬럼 마스킹만으로는 대각선이라 못 걸러짐). 사다리꼴 밖을 워프
        #   전에 검게 지우면, BEV_SRC가 정확히 BEV_DST 사각형으로 매핑되도록 풀린
        #   호모그래피 특성상 bev에서도 그 밖은 항상 수직 경계로 깔끔하게 검게
        #   나와서 valid_x 마스킹으로 완전히 제거된다.
        trapezoid_mask = np.zeros((self.roi_h, self.roi_w), dtype=np.uint8)
        cv2.fillConvexPoly(trapezoid_mask, bev_src.astype(np.int32), 255)
        roi_for_warp = cv2.bitwise_and(self.roi, self.roi, mask=trapezoid_mask)

        M = cv2.getPerspectiveTransform(
            bev_src, bev_dst
        )
        self.bev = cv2.warpPerspective(
            roi_for_warp,
            M,
            (self.roi_w, self.roi_h)
        )

        # white Lane : Adaptive Thresholding
        # 1) Gray 변환
        gray = cv2.cvtColor(self.bev, cv2.COLOR_BGR2GRAY)
        # 2) CLAHE (국소 명암 향상) — adaptiveThreshold 전에 대비를 한 번 더 올려줌
        clahe = cv2.createCLAHE(
            clipLimit=1.5, #1.2
            tileGridSize=(8,8)
        )
        gray = clahe.apply(gray)

        # 3) Median Blur — 폭이 얇은 반사 줄무늬 제거(굵은 실제 차선은 보존)
        gray = cv2.medianBlur(gray, MEDIAN_BLUR_KSIZE)

        # 4) Adaptive Thresholding — 픽셀마다 주변 blockSize 영역의 가우시안 가중평균
        #    밝기를 기준으로 이진화(THRESH_BINARY: 평균-C 보다 밝으면 흰색)
        self.white = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            ADAPTIVE_BLOCK_SIZE, ADAPTIVE_C
        )

        # 5) Canny Edge Detection — 밝기 기준(adaptiveThreshold)이 놓치는 저대비
        #    차선 경계를 그레이디언트 기준으로 보완, 합집합(OR)으로 합쳐 재현율을 높인다
        blurred = cv2.GaussianBlur(
            gray, (GAUSSIAN_BLUR_KSIZE, GAUSSIAN_BLUR_KSIZE), 0
        )
        self.edges = cv2.Canny(blurred, CANNY_LOW, CANNY_HIGH)
        self.white = cv2.bitwise_or(self.white, self.edges)

        # HSV
        hsv = cv2.cvtColor(self.bev, cv2.COLOR_BGR2HSV)

        #Yellow Lane
        self.yellow = cv2.inRange(
            hsv, np.array([15,80,80]), np.array([40,255,255])
        )

        # BEV 워프 유효 영역 밖(원근변환 사다리꼴 바깥) 마스킹
        #   BEV_DST가 정의하는 x범위(169~489/640 비율) 밖은 실제 도로가 아니라
        #   getPerspectiveTransform이 원근평면을 무한히 확장해서 만들어낸 외삽
        #   픽셀이다(실측: 텍스처 없는 균일한 바닥에서도 이 바깥 테두리가
        #   CLAHE+adaptiveThreshold를 통과해 흰색으로 오검출됨 — 게다가 바로
        #   좌/우 절반 탐색범위 안쪽 끝에 걸쳐있어 차선 오판으로 이어지기 쉬움).
        #   BEV_DST 비율로 유효 x범위를 계산해 그 밖은 흰/노랑 마스크에서 지운다.
        MARGIN = 20

        valid_x_lo = int(BEV_DST[0, 0] * self.roi_w - MARGIN)
        valid_x_hi = int(BEV_DST[1, 0] * self.roi_w + MARGIN)
        self.white[:, :valid_x_lo] = 0
        self.white[:, valid_x_hi:] = 0
        self.yellow[:, :valid_x_lo] = 0
        self.yellow[:, valid_x_hi:] = 0

        #Morphology
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (3,3)
        )

        # white cclosing
        self.white = cv2.morphologyEx(
            self.white, cv2.MORPH_CLOSE, kernel
        )


        # Connected Components — 작은 반사광 얼룩/잡음 제거(면적 기준)
        num, labels, stats, _ = cv2.connectedComponentsWithStats(self.white)

        mask = np.zeros_like(self.white)

        for i in range(1, num) :
            area = stats[i, cv2.CC_STAT_AREA]

            if area > WHITE_CCA_MIN_AREA :
                mask[labels == i] = 255

        self.white = mask

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

        # 흰색 마스크에서 노란색 확정 영역 제외
        #   adaptiveThreshold는 그레이스케일(밝기)만 보기 때문에 채도가 높은
        #   노란 중앙선도 "주변보다 밝다"는 조건을 만족해버려 self.white에 같이
        #   잡힌다(실측으로 확인됨). self.yellow는 이미 색상 기반으로 노란색만
        #   따로 걸러낸 마스크이므로, 여기 확정된 자리는 흰색일 수 없다는 관계를
        #   그대로 강제한다. dilate로 살짝 여유를 줘서 노란선 가장자리의 안티
        #   앨리어싱된 픽셀까지 같이 제외한다.
        yellow_dilated = cv2.dilate(self.yellow, kernel)
        self.white = cv2.bitwise_and(
            self.white, cv2.bitwise_not(yellow_dilated)
        )

        #DEBUG
        if DEBUG_VIZ_LANE:
            cv2.imshow("lane_bev", self.bev)
            cv2.imshow("lane_edges", self.edges)
            cv2.imshow("lane_white", self.white)
            cv2.imshow("lane_yellow", self.yellow)

        return (self.bev, self.white, self.yellow)

class SlideWindow:
    """
    이름은 track_drive.py와의 인터페이스 호환을 위해 SlideWindow로 유지하지만,
    내부 구현은 "히스토그램 슬라이딩 윈도우 + 2차 polyfit + 이전 프레임 기반 탐색"
    대신 "구간별 무게중심(Moments)" 방식으로 교체됐다. ROI를 위아래로 나눠
    구간마다 cv2.moments()로 무게중심만 구하고 그 점들을 그대로 잇기 때문에
    폴리피팅이 필요 없고, 반사광으로 특정 구간이 오염돼도 다른 구간엔 영향이
    없어 커브 대응이 더 직관적이다.
    """
    def __init__(self, *, n_slices=None, min_pixels=None, near_slices=None, far_slices=None,
                 slice_outlier_max=None, slice_fit_min=None, stable_frame_min=None,
                 stable_jump_max=None, lane_width_init=None, lane_width_min=None,
                 lane_width_max=None, path_n_waypoints=None):
        self.vis = None

        self.roi_h = 0
        self.roi_w = 0

        # 튜닝 상수를 모듈 전역 대신 인스턴스 속성으로 캡처해둔다(기본값은 기존 모듈 상수와
        # 동일 — classic 파이프라인 동작은 그대로 유지). DL 기반 서브클래스(dl_lane.py의
        # DLSlideWindow)처럼 좌표 스케일이 다른 파생 클래스가 생성자 인자로 자기 스케일에
        # 맞는 값을 넣을 수 있도록 파라미터화한 것뿐, 알고리즘 자체는 바뀌지 않는다.
        self.n_slices = n_slices if n_slices is not None else MOMENT_N_SLICES
        self.min_pixels = min_pixels if min_pixels is not None else MOMENT_MIN_PIXELS
        self.near_slices = near_slices if near_slices is not None else MOMENT_NEAR_SLICES
        self.far_slices = far_slices if far_slices is not None else MOMENT_FAR_SLICES
        self.slice_outlier_max = slice_outlier_max if slice_outlier_max is not None else LANE_SLICE_OUTLIER_MAX
        self.slice_fit_min = slice_fit_min if slice_fit_min is not None else LANE_SLICE_FIT_MIN
        self.stable_frame_min = stable_frame_min if stable_frame_min is not None else STABLE_FRAME_MIN
        self.stable_jump_max = stable_jump_max if stable_jump_max is not None else STABLE_JUMP_MAX
        self.lane_width_min = lane_width_min if lane_width_min is not None else 180.0
        self.lane_width_max = lane_width_max if lane_width_max is not None else 400.0

        #실시간 차선폭 (한쪽 차선만 보일 때 반대쪽 추정에 사용)
        self.lane_width = lane_width_init if lane_width_init is not None else 260.0

        # 구간별 무게중심 저장 — calc_center()/시각화에서 재사용
        #   각 리스트는 길이 self.n_slices, 인덱스 0=가장 아래(근거리) ~ 마지막=가장 위(원거리)
        self.left_centers = []
        self.right_centers = []
        self.yellow_centers = []

        # 명시적 경로(웨이포인트) — controller/pure_pursuit.py가 소비
        #   [(x,y), ...] ROI 픽셀좌표, 인덱스 0=차량과 가장 가까운 점 ~ 마지막=가장 먼 점.
        #   calc_center()가 매 프레임 갱신하되, 이번 프레임에 유효 슬라이스가 2개 미만이면
        #   갱신하지 않고 직전 값을 그대로 유지한다(perc_lane()이 lane_offset을 무효 프레임에
        #   직전 EMA값으로 유지하는 것과 같은 원칙 — 폴백은 "값을 0으로 지우지 않고 유지").
        self.path_n_waypoints = path_n_waypoints if path_n_waypoints is not None else PATH_N_WAYPOINTS
        self.path = []

        # 프레임 간 스파이크 필터링(디바운스) 상태 — _debounce()에서 사용
        self._confirmed = None    # 마지막으로 확정되어 실제 출력되는 (valid, offset, lookahead, center)
        self._pending = None      # 연속성 검사 중인 후보값
        self._pending_count = 0   # 후보가 연속으로 유지된 프레임 수

    def _slice_centers(self, mask, x_offset, color, min_pixels=None):
        """
        mask를 아래→위로 MOMENT_N_SLICES개 구간으로 나눠 구간마다 cv2.moments()로
        무게중심(cx)을 구한다. 슬라이딩 윈도우처럼 이전 구간 위치를 이어받지
        않고 구간마다 독립적으로 계산하므로, 한 구간이 반사/잡음으로 날아가도
        다른 구간에는 전혀 영향이 없다.
          입력 : mask       — 이진마스크(좌/우 절반 등으로 이미 열 방향이 잘려있을 수 있음)
                x_offset   — mask가 원본 ROI에서 잘려나온 경우의 x좌표 보정값(좌우 분리용)
                color      — 디버그 사각형 색상
                min_pixels — 구간 채택 최소 픽셀수. None이면 self.min_pixels(원래 da용
                             기본값)를 쓴다. da처럼 면을 채우는 마스크가 아니라 ll처럼
                             가늘게 선만 있는 마스크에 쓸 땐 그 선 전용 임계값(예:
                             DL_LL_SIDE_MIN_PIXELS)을 넘겨서 self.min_pixels(da 기준,
                             훨씬 큼)로 인해 항상 무효 처리되는 걸 막아야 한다
                             (DLSlideWindow.detect()의 yellow_band_centers 계산 참고).
          출력 : centers — 길이 self.n_slices 리스트. 각 원소는 (y_center, cx) 또는
                 해당 구간 픽셀수가 min_pixels 미만이면 None
        """
        slice_h = self.roi_h // self.n_slices
        centers = []
        threshold = self.min_pixels if min_pixels is None else min_pixels

        for i in range(self.n_slices):
            y_high = self.roi_h - i * slice_h
            # 마지막(가장 위) 구간은 정수나눗셈 나머지를 포함해 y=0까지 전부 커버
            y_low = 0 if i == self.n_slices - 1 else self.roi_h - (i + 1) * slice_h

            band = mask[y_low:y_high, :]

            if DEBUG_VIZ_LANE:
                cv2.rectangle(
                    self.vis,
                    (x_offset, y_low), (x_offset + mask.shape[1] - 1, max(y_high - 1, y_low)),
                    color, 1
                )

            M = cv2.moments(band, binaryImage=True)

            if M['m00'] < threshold:
                centers.append(None)
                continue

            cx = M['m10'] / M['m00'] + x_offset
            y_center = (y_low + y_high) / 2.0
            centers.append((y_center, cx))

        return centers

    def _slice_edge_midpoints(self, mask, x_offset, color=None):
        """_slice_centers()와 같은 밴드 분할이지만, 무게중심(moments) 대신 "밴드 내
        가장 왼쪽 픽셀 열 + 가장 오른쪽 픽셀 열의 중점"을 밴드 중심으로 쓴다.

        [2026-08-07] da(주행가능영역)처럼 픽셀이 넓게 채워진 마스크는, 갓길 등 한쪽
        여백이 넓으면 무게중심(픽셀 밀도 가중 평균)이 그 여백 쪽으로 쏠려 실제 도로
        폭의 "정중앙"에서 벗어나는 문제가 있다 — Voronoi(좌우 경계에서 등거리인 점)
        아이디어에서 착안했다. 진짜 Voronoi/skeletonization을 계산할 필요 없이,
        밴드마다 좌/우 끝 픽셀 열의 중점만 써도 같은 효과를 훨씬 싸게 얻는다 — 밀도가
        아니라 순수 "폭의 중앙"을 잡으므로 여백 크기와 무관하다. da 전용으로 만들었다
        (ll/좌우차선처럼 원래 얇은 선은 두 방식의 차이가 미미해 그대로 _slice_centers를
        쓴다).
          입력/출력 형식은 _slice_centers()와 동일 — centers[i] = (y_center, cx) 또는
          해당 구간 픽셀수가 self.min_pixels 미만이면 None.
        """
        slice_h = self.roi_h // self.n_slices
        centers = []

        for i in range(self.n_slices):
            y_high = self.roi_h - i * slice_h
            y_low = 0 if i == self.n_slices - 1 else self.roi_h - (i + 1) * slice_h

            band = mask[y_low:y_high, :]

            if DEBUG_VIZ_LANE and color is not None:
                cv2.rectangle(
                    self.vis,
                    (x_offset, y_low), (x_offset + mask.shape[1] - 1, max(y_high - 1, y_low)),
                    color, 1
                )

            if cv2.countNonZero(band) < self.min_pixels:
                centers.append(None)
                continue

            cols = np.nonzero(np.any(band > 0, axis=0))[0]
            cx = (float(cols[0]) + float(cols[-1])) / 2.0 + x_offset
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
        if len(valid_idx) < self.slice_fit_min:
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

            if residual > self.slice_outlier_max:
                result[i] = None
                if DEBUG_VIZ_LANE:
                    cv2.drawMarker(
                        self.vis,
                        (int(np.clip(x_i, 0, self.roi_w - 1)), int(y_i)),
                        (0, 0, 255), cv2.MARKER_TILTED_CROSS, 12, 2
                    )

        return result

    def _group_mean(self, centers, count, from_start):
        """
        centers(길이 MOMENT_N_SLICES, 아래→위 순)에서 근거리(from_start=True,
        앞쪽 count개) 또는 원거리(from_start=False, 뒤쪽 count개) 구간들의 cx
        평균을 낸다. 여러 구간을 평균내서 구간 하나의 노이즈에 결과가 흔들리는
        걸 줄인다. 유효한 구간이 하나도 없으면 None.
        """
        window = centers[:count] if from_start else centers[-count:]
        vals = [c[1] for c in window if c is not None]

        if not vals:
            return None
        return float(np.mean(vals))

    def draw_centers(self, centers, color):
        """구간별 무게중심을 점으로 찍고 인접한 점끼리 이어서 커브 형태로 시각화."""
        pts = [
            (int(np.clip(cx, 0, self.roi_w - 1)), int(y))
            for c in centers if c is not None
            for (y, cx) in [c]
        ]

        for pt in pts:
            cv2.circle(self.vis, pt, 4, color, -1)

        for p1, p2 in zip(pts, pts[1:]):
            cv2.line(self.vis, p1, p2, color, 2)

    def draw_path(self, path):
        """피팅된 명시적 경로(웨이포인트)를 자홍색 폴리라인으로 시각화.
        left/right/yellow_centers(노란/청록 점)와 색을 겹치지 않게 구분해서,
        "슬라이스별 원본 관측점"과 "그걸 피팅해 만든 최종 경로"를 한눈에 대조할 수 있게 한다."""
        if not path:
            return
        pts = [
            (int(np.clip(x, 0, self.roi_w - 1)), int(np.clip(y, 0, self.roi_h - 1)))
            for x, y in path
        ]
        for p1, p2 in zip(pts, pts[1:]):
            cv2.line(self.vis, p1, p2, (255, 0, 255), 2)
        for p in pts:
            cv2.circle(self.vis, p, 3, (255, 0, 255), -1)

    def visualize(self, offset):
        self.draw_centers(self.left_centers, (0,255,255))
        self.draw_centers(self.right_centers, (0,255,255))
        self.draw_centers(self.yellow_centers, (0,165,255))
        self.draw_path(self.path)

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

    def _build_centerline_points(self):
        """슬라이스별(구간별) 차선 중심점을 만든다 — calc_center()의 근거리/원거리
        "그룹 평균" 로직과 같은 4단계 폴백을 슬라이스 하나하나에 적용한 버전이다.
          1. 양쪽 다 검출 + 폭이 정상범위  → 중점
          2. 왼쪽만 검출                  → 왼쪽 + lane_width/2 (EMA 추적값, calc_center()가
                                            이미 이번 프레임 값으로 갱신해둔 것을 그대로 씀)
          3. 오른쪽만 검출                → 오른쪽 - lane_width/2
          4. 흰 차선 둘 다 없음           → 노란 중심선 그대로 사용
          양쪽은 검출됐는데 폭이 비정상이면 그 슬라이스는 스킵한다(반사광 등으로 한쪽이
          오검출됐을 가능성이 높은데 슬라이스 하나만으로는 어느 쪽인지 판단 근거가 없음 —
          calc_center()가 이 경우 프레임 전체를 무효 처리하는 것과 같은 이유, 다만 여기서는
          슬라이스 단위라 그 슬라이스만 빼고 나머지로 곡선을 피팅할 수 있다).
        출력 : [(y, cx), ...] 유효한 슬라이스만, 순서는 self.left_centers와 동일(근거리→원거리).
        """
        points = []
        for i in range(self.n_slices):
            l = self.left_centers[i] if i < len(self.left_centers) else None
            r = self.right_centers[i] if i < len(self.right_centers) else None

            if l is not None and r is not None:
                width = r[1] - l[1]
                if not (self.lane_width_min < width < self.lane_width_max):
                    continue
                y_ref, cx = (l[0] + r[0]) / 2.0, (l[1] + r[1]) / 2.0
            elif l is not None:
                y_ref, cx = l[0], l[1] + self.lane_width / 2.0
            elif r is not None:
                y_ref, cx = r[0], r[1] - self.lane_width / 2.0
            else:
                yc = self.yellow_centers[i] if i < len(self.yellow_centers) else None
                if yc is None:
                    continue
                y_ref, cx = yc

            points.append((y_ref, cx))
        return points

    def _fit_and_sample_path(self, points):
        """슬라이스 중심점을 구간별 선형보간(np.interp)으로 이어, 차량 근처(roi_h)부터
        가장 먼 유효 슬라이스까지 균등 간격으로 self.path_n_waypoints개를 샘플링한다.
        점이 2개 미만이면(선조차 정의 못함) None — 호출부(calc_center())가 직전 경로를
        유지하는 폴백을 담당한다.

        [2026-08-07] 원래는 점 3개 이상이면 2차 다항식(커브 대응) 하나로 전 구간을
        피팅하고, roi_h(차량 바로 앞)까지 그 곡선으로 외삽했다. 슬라이스 3~5개짜리
        저차수 피팅이라 노이즈로 계수가 조금만 튀어도, 그 외삽 구간(실측 데이터가
        하나도 없는 근거리 — 조향에 가장 큰 영향을 주는 구간)이 크게 휘어지는 문제가
        있었다(PATH_EXTRAPOLATE_MARGIN 클램프는 안전판일 뿐 흔들림 자체를 줄이진 못함).
        참고 프로젝트(github.com/junhyukch7/Advanced-Lane-Detection)가 같은 이유로
        고차 다항식 대신 슬라이딩 윈도우 점들을 직선으로 잇는 선형보간을 쓴 걸 보고
        동일하게 바꿨다. np.interp는 데이터 범위 밖에서 곡선으로 외삽하지 않고 가장
        가까운 실측점의 x를 그대로 유지(hold)하므로, roi_h까지의 근거리 구간에 실측
        밴드가 없어도 값이 곡선으로 튈 수가 없다 — 외삽 자체가 없어져 클램프보다
        근본적으로 안전하다. 실차 미검증.
        """
        if len(points) < 2:
            return None

        ys = np.array([p[0] for p in points], dtype=np.float64)
        xs = np.array([p[1] for p in points], dtype=np.float64)

        y_near = float(self.roi_h)
        y_far = float(np.min(ys))
        if y_near <= y_far:
            return None

        order = np.argsort(ys)  # np.interp는 xp가 오름차순이어야 함
        ys_sorted = ys[order]
        xs_sorted = xs[order]

        sample_ys = np.linspace(y_near, y_far, self.path_n_waypoints)
        sample_xs = np.interp(sample_ys, ys_sorted, xs_sorted)

        return [(float(x), float(y)) for x, y in zip(sample_xs, sample_ys)]

    def _update_path(self, fitted_path):
        """새로 피팅된 경로(fitted_path)를 직전 self.path와 웨이포인트 인덱스별로
        EMA 블렌딩해서 반영한다(모듈 상단 PATH_EMA_ALPHA 주석 참고). fitted_path가
        None이면(유효 슬라이스 2개 미만) 아무것도 안 하고 직전 경로를 그대로 둔다 —
        기존 "무효 프레임엔 마지막 값 유지" 원칙과 동일. 직전 경로가 비어있거나
        (첫 프레임) 길이가 다르면(PATH_N_WAYPOINTS 변경 등 비정상 상황) 블렌딩할
        기준이 없으므로 새 경로를 그대로 채택한다."""
        if fitted_path is None:
            return

        if not self.path or len(self.path) != len(fitted_path):
            self.path = fitted_path
            return

        a = PATH_EMA_ALPHA
        self.path = [
            (a * nx + (1.0 - a) * px, a * ny + (1.0 - a) * py)
            for (px, py), (nx, ny) in zip(self.path, fitted_path)
        ]

    def calc_center(self):
        near_left   = self._group_mean(self.left_centers,   self.near_slices, True)
        far_left    = self._group_mean(self.left_centers,   self.far_slices,  False)
        near_right  = self._group_mean(self.right_centers,  self.near_slices, True)
        far_right   = self._group_mean(self.right_centers,  self.far_slices,  False)
        near_yellow = self._group_mean(self.yellow_centers, self.near_slices, True)
        far_yellow  = self._group_mean(self.yellow_centers, self.far_slices,  False)

        lane_valid = False
        offset = 0
        lookahead = 0
        near_center = far_center = None

    # 1. 양쪽 차선 모두 검출
        if near_left is not None and near_right is not None:

            width = near_right - near_left

            if self.lane_width_min < width < self.lane_width_max:
                alpha = 0.1
                self.lane_width = (
                   (1 - alpha) * self.lane_width +
                    alpha * width
                )

                near_center = (near_left + near_right) / 2
                far_center = (
                    (far_left  if far_left  is not None else near_left) +
                    (far_right if far_right is not None else near_right)
                ) / 2
                lane_valid = True
            # width가 정상 범위 밖이면 left/right 중 하나가 반사·오검출일 가능성이
            # 높은데, 이 두 값만으로는 어느 쪽이 진짜 차선인지 판단할 근거가 없다.
            # 잘못 고른 쪽으로 offset을 내면 오히려 위험하므로 이번 프레임은
            # lane_valid=False(무효)로 남겨 상위 로직이 직진/정속으로 폴백하게 한다.

    # 2. 왼쪽 차선만 검출
        elif near_left is not None:

            far_ref = far_left if far_left is not None else near_left
            near_center = near_left + self.lane_width / 2
            far_center = far_ref + self.lane_width / 2
            lane_valid = True

    # 3. 오른쪽 차선만 검출
        elif near_right is not None:

            far_ref = far_right if far_right is not None else near_right
            near_center = near_right - self.lane_width / 2
            far_center = far_ref - self.lane_width / 2
            lane_valid = True

    # 4. 흰 차선을 전혀 못 찾았을 때만 노란 차선 사용
        elif near_yellow is not None:

            far_ref = far_yellow if far_yellow is not None else near_yellow
            near_center = near_yellow
            far_center = far_ref
            lane_valid = True

        if lane_valid:
            offset = near_center - self.roi_w / 2
            lookahead = far_center - self.roi_w / 2

        lane_center = self.roi_w / 2 + offset

        # 명시적 경로(웨이포인트) — self.lane_width가 위에서 이번 프레임 값으로 이미
        # 갱신됐으므로(양쪽 검출+폭 정상 분기), _build_centerline_points()의 한쪽-차선
        # 폴백이 최신 추정 폭을 쓴다. 유효 슬라이스가 2개 미만이면 fitted_path가 None이
        # 되고, 그 경우 _update_path()가 self.path를 갱신하지 않아(직전 프레임 값 유지)
        # offset/lane_offset과 동일한 "무효 프레임엔 마지막 값 유지" 원칙을 따른다.
        # 유효할 때도 그대로 대입하지 않고 직전 경로와 EMA 블렌딩한다(PATH_EMA_ALPHA
        # 주석 참고) — 조향이 매 프레임 새로 피팅된 경로에 과민하게 반응하는 걸 막기 위함.
        fitted_path = self._fit_and_sample_path(self._build_centerline_points())
        self._update_path(fitted_path)

        lane_valid, offset, lookahead, lane_center = self._debounce(
            lane_valid, offset, lookahead, lane_center
        )

        self.visualize(offset)

        return lane_valid, offset, lookahead, lane_center, self.path

    def _debounce(self, valid, offset, lookahead, center):
        """
        프레임 간 스파이크 필터링. 이번 프레임 결과가 직전 "후보"와
        self.stable_jump_max(px) 이내로 비슷하면 후보 연속 프레임 수를 늘리고,
        벗어나면 새 후보로 교체하며 카운트를 1로 리셋한다. 후보가
        self.stable_frame_min 프레임 연속으로 유지돼야만 확정값으로 승격되어
        실제로 반환된다 — 승격 전까지는 마지막 확정값을 그대로 반환하므로
        1~2프레임짜리 튐이 조향에 바로 반영되지 않는다.
        """
        candidate = (valid, offset, lookahead, center)

        if self._confirmed is None:
            self._confirmed = candidate
            self._pending = candidate
            self._pending_count = self.stable_frame_min
            return candidate

        same_flow = (
            valid == self._pending[0] and
            abs(offset - self._pending[1]) <= self.stable_jump_max
        )

        if same_flow:
            self._pending_count += 1
        else:
            self._pending = candidate
            self._pending_count = 1

        if self._pending_count >= self.stable_frame_min:
            self._confirmed = self._pending

        return self._confirmed

    def detect(self, bev, white, yellow):
        self.roi_h, self.roi_w = white.shape
        self.vis = bev.copy()

        # 폭을 5등분해서 가운데 1/5(중앙선 자리)는 노란색 전용, 좌/우 나머지
        # 2/5씩은 흰색 전용으로 나눈다 — 이전엔 흰색 좌/우 절반과 노란색 중앙
        # 탐색범위(quarter)가 서로 겹쳐서 경계 부근 픽셀을 두 마스크가 같이
        # 훑었다. 이제 세 구간이 겹치지 않게 나눠 각자의 색만 본다.
        fifth = self.roi_w // 5
        mid_lo = fifth * 2
        mid_hi = fifth * 3

        # 좌/우로 나눠 각각 구간별 무게중심 계산 →
        # 구간 간 일관성 체크로 반사광 등 이상치 슬라이스 제외
        self.left_centers = self._reject_outliers(self._slice_centers(
            white[:, :mid_lo], 0, (0,255,0)
        ))
        self.right_centers = self._reject_outliers(self._slice_centers(
            white[:, mid_hi:], mid_hi, (0,255,0)
        ))
        self.yellow_centers = self._reject_outliers(self._slice_centers(
            yellow[:, mid_lo:mid_hi], mid_lo, (0,180,255)
        ))

        return self.calc_center()
