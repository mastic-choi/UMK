import numpy as np
import cv2

#SlidingWindow
# 출처: 전전년도 타 팀 코드(KUAC_2024-main) lane_detection/src/slidewindow_both_lane.py
#   nwindows=14, margin=20, minpix=10 을 그대로 초기값으로 이식 (실차 미검증, 튜닝 전제)
SW_NWIN = 14
SW_MARGIN = 20
SW_MINPIX = 10
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
LANE_LOOKAHEAD = 0.35  # KUAC_2024엔 예측조향 개념 자체가 없어 대응값 없음 — 기존값 유지
# Yellow Lane
LANE_YELLOW_WEIGHT = 0.25
LANE_YELLOW_MAX_DEV = 40
#Debug
DEBUG_VIZ_LANE = True

# Canny + Hough 반사광(Glare) 필터
#   문제 : 트랙 바닥 정반사가 CLAHE+Top-Hat+HSV 조건(밝고 흐릿한 흰색)을 흰 차선과
#         똑같이 만족해버려서 슬라이딩 윈도우가 반사광 얼룩을 차선으로 오검출함.
#   해결 : BEV 위에서 Canny 엣지 → HoughLinesP로 "직선 형태"만 추출해서 짧은
#         파편(반사광 얼룩 경계 조각 등)을 제거한 뒤, 기존 색상/명암 마스크와
#         AND로 결합한다. "밝고(색상 조건) + 선분 모양(형태 조건)"을 모두
#         만족하는 픽셀만 최종 흰 차선으로 인정.
#   ※ 각도(수직 근방) 필터는 제거했다 — 커브가 심한 구간에서 화면 위쪽(먼 거리)
#     차선 선분이 수직에서 크게 벗어나(거의 눕는 각도) 잘려나가 커브 추종이
#     깨지는 문제가 있었음. 대신 SlideWindow의 이전 프레임 기반 탐색
#     (prior-based search, 아래 LANE_PRIOR_* 참고)이 반사광 방어를 주로 담당하고,
#     여기 Hough 필터는 "길이 필터"만으로 짧은 파편 제거 역할만 한다.
#   실차 미검증 튜닝값 — 반사광이 계속 새면 임계값을, 점선차선이 끊기면
#   MIN_LINE_LEN/MAX_LINE_GAP을 먼저 재조정할 것.
CANNY_BLUR_KSIZE      = 5      # 가우시안 블러 커널(홀수). 너무 크면 가는 점선 차선까지 뭉개짐
CANNY_LOW, CANNY_HIGH = 50, 150   # Canny 이력임계값(하/상). BEV가 CLAHE로 이미 대비를 올려놔서 낮게 잡음
HOUGH_RHO             = 1             # 거리 해상도(px)
HOUGH_THETA           = np.pi / 180   # 각도 해상도(rad)
HOUGH_THRESHOLD       = 20            # 직선으로 인정할 최소 누적표(투표) 수
HOUGH_MIN_LINE_LEN    = 15            # 최소 선분 길이(px) — 반사광 얼룩의 짧은 파편 제거용
HOUGH_MAX_LINE_GAP    = 15            # 같은 직선으로 이어붙일 최대 틈(px) — 점선차선 조각 연결
HOUGH_LINE_THICKNESS  = 7    # 재구성 마스크에 그릴 선 두께(px) — 슬라이딩윈도우 margin(20px)보다 작게

# 이전 프레임 기반 탐색 (prior-based / look-ahead search)
#   문제 : 매 프레임 히스토그램으로 처음부터 다시 찾으면, 반사광이 순간적으로
#         차선보다 밝게 찍혔을 때 히스토그램 최댓값 자리를 반사가 차지해버려
#         바로 그쪽을 잘못 추종함. 밝기/모양으로는 반사광과 차선이 이 트랙에서
#         잘 안 갈라진다는 게 확인됐음(HSV·Hough 각도 모두 시도 후 결론).
#   해결 : 직전 프레임에 유효했던 fit(2차 곡선) 근방 margin(px) 안에서만
#         픽셀을 모은다. 진짜 차선은 프레임 간 위치가 부드럽게 이어지지만,
#         반사광은 엉뚱한 위치에서 순간적으로 나타났다 사라지는 경향이 있어
#         애초에 탐색범위 밖이라 무시됨 — 색상/형태와 무관한 시간적 일관성 필터.
#   fit이 없거나(첫 프레임) LANE_PRIOR_MISS_MAX 프레임 연속 실패하면 그때만
#   histogram()+sliding_window()로 전체 재탐색(폴백)한다.
LANE_PRIOR_MARGIN   = 30   # 이전 fit 곡선 좌우로 탐색할 폭(px). SW_MARGIN(20)보다 살짝 넓게
LANE_PRIOR_MISS_MAX = 5    # 이 프레임 수 연속 실패하면 fit 폐기 + 전체 히스토그램 재탐색으로 폴백


#def Debugging(flag):
# DEBUG 상수들 모아놓자(외부 파일로 뺄지는 고민해보기)
class CameraProcessor:
    def __init__(self):
        self.roi = None
        self.bev = None
        self.white = None
        self.yellow = None

        self.roi_h = 0
        self.roi_w = 0

    def _hough_line_mask(self, gray):
        """
        Canny 엣지 → HoughLinesP 선분 추출 → 길이 필터링 → 살아남은
        선분만 굵게 재구성해서 '차선 모양(직선)' 이진 마스크를 만든다.
          입력 : gray  — CLAHE까지 적용된 BEV 그레이스케일(호출부에서 재사용, 중복연산 방지)
          출력 : hough_mask — gray와 동일 크기의 0/255 이진 마스크
        각도 필터는 두지 않는다(커브 심한 구간에서 차선 선분이 수직에서 크게
        벗어나 잘려나가는 문제 때문에 제거함) — 반사광 방어는 이제 주로
        SlideWindow의 이전 프레임 기반 탐색(prior-based search)이 담당한다.
        """
        # 1) 가우시안 블러 — Canny 전에 고주파 잡음(반사광 얼룩의 거친 경계) 완화
        blur = cv2.GaussianBlur(
            gray, (CANNY_BLUR_KSIZE, CANNY_BLUR_KSIZE), 0
        )

        # 2) Canny 엣지 검출
        edges = cv2.Canny(blur, CANNY_LOW, CANNY_HIGH)

        # 3) 확률적 허프 변환 — 엣지 중 "직선 구간"만 (x1,y1,x2,y2) 선분으로 추출
        lines = cv2.HoughLinesP(
            edges,
            HOUGH_RHO,
            HOUGH_THETA,
            HOUGH_THRESHOLD,
            minLineLength=HOUGH_MIN_LINE_LEN,
            maxLineGap=HOUGH_MAX_LINE_GAP
        )

        hough_mask = np.zeros_like(gray)

        if lines is None:
            return hough_mask

        # 4) 길이 필터링 후 살아남은 선분만 재구성 마스크에 그림 (각도 무관)
        for line in lines:
            x1, y1, x2, y2 = line[0]

            # 길이 필터 : 짧은 파편(반사광 얼룩 경계 등) 제거
            length = float(np.hypot(x2 - x1, y2 - y1))
            if length < HOUGH_MIN_LINE_LEN:
                continue

            cv2.line(
                hough_mask, (x1, y1), (x2, y2),
                255, HOUGH_LINE_THICKNESS
            )

        return hough_mask

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

        # white Lane : Local Contrast
        # 1) Gray 변환
        gray = cv2.cvtColor(self.bev, cv2.COLOR_BGR2GRAY)
        # 2) CLAHE (극소 명암 향상)
        clahe = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8,8)
        )
        gray = clahe.apply(gray)

        # 3) Top-Hat(극소 대비)
        kernel_tophat = cv2.getStructuringElement(
            cv2.MORPH_RECT, (21,21)
        )

        contrast = cv2.morphologyEx(
            gray, cv2.MORPH_TOPHAT, kernel_tophat
        )

        #Threshold
        _, white_tophat = cv2.threshold(
            contrast, 10, 255, cv2.THRESH_BINARY
        )

        # HSV
        hsv = cv2.cvtColor(self.bev, cv2.COLOR_BGR2HSV)

        # white HSV Mask
        white_hsv = cv2.inRange(
            hsv, np.array([0,0,150]), np.array([180,90,255])
        )

        # Top-Hat + HSV 결합 (색상/명암 조건 — "밝고 흰색인가")
        white_color = cv2.bitwise_and(
            white_tophat, white_hsv
        )

        # Canny + Hough 직선 마스크 (형태 조건 — "직선 모양인가")
        #   CLAHE까지 적용된 gray를 그대로 재사용(cvtColor 중복 호출 방지)
        hough_mask = self._hough_line_mask(gray)

        # 색상 조건 AND 형태 조건 → 둘 다 만족하는 픽셀만 흰 차선으로 인정
        # (반사광 얼룩은 색상 조건은 통과해도 형태(직선) 조건에서 대부분 걸러짐)
        self.white = cv2.bitwise_and(
            white_color, hough_mask
        )

        #Yellow Lane
        self.yellow = cv2.inRange(
            hsv, np.array([15,80,80]), np.array([40,255,255])
        )
        #Morphology
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (3,3)
        )

        # white cclosing
        self.white = cv2.morphologyEx(
            self.white, cv2.MORPH_CLOSE, kernel
        )


        # Connected Components
        num, labels, stats, _ = cv2.connectedComponentsWithStats(self.white)

        mask = np.zeros_like(self.white)

        for i in range(1, num) :
            area = stats[i, cv2.CC_STAT_AREA]

            if area> 5 :
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

        #DEBUG
        if DEBUG_VIZ_LANE:
            cv2.imshow("lane_bev", self.bev)

            # BEV(위에서 내려다본 카메라 화면) + Canny/Hough로 "차선"이라 추정한
            # 형태 마스크를 한 창에 나란히 붙여서 비교 — 반사광이 걸러지는지 한눈에 확인용
            hough_vis = cv2.cvtColor(hough_mask, cv2.COLOR_GRAY2BGR)
            hough_debug = cv2.hconcat([self.bev, hough_vis])
            cv2.imshow("lane_hough_debug", hough_debug)

            cv2.imshow("lane_white", self.white)
            cv2.imshow("lane_yellow", self.yellow)

        return (self.bev, self.white, self.yellow)
    
class SlideWindow:
    def __init__(self):
        self.left_base = -1
        self.right_base = -1

        self.left_idx = None
        self.right_idx = None

        self.left_fit = None
        self.right_fit = None

        self.yellow_fit = None
        
        self.vis = None

        self.roi_h = 0
        self.roi_w = 0

        #실시간 차선폭
        self.lane_width = 260.0

        # 이전 프레임 기반 탐색(prior-based search) 연속 실패 카운터
        #   fit이 갱신되면 0으로 리셋, 실패할 때마다 +1. LANE_PRIOR_MISS_MAX
        #   넘으면 fit을 폐기하고 다음 프레임에 전체 히스토그램 재탐색으로 폴백.
        self._left_miss_cnt = 0
        self._right_miss_cnt = 0
        self._yellow_miss_cnt = 0

    def visualize(self, offset):
        self.draw_fit(self.left_fit, (0,255,255))
        self.draw_fit(self.right_fit, (0,255,255))
        self.draw_fit(self.yellow_fit, (0,165,255))

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

    def histogram(self, mask, search=None):
        hist = np.sum(mask[int(self.roi_h*0.65):, :], axis=0)

        if search is None:
            hist_part = hist
            offset = 0
        else:
            lo, hi = search
            hist_part = hist[lo:hi]
            offset = lo

        if hist_part.max() == 0:
            return -1
        
        return int(np.argmax(hist_part)+offset)
    
    def fit_curve(self, idx, y, x, threshold):
        if len(idx) < threshold:
            return None
        
        return np.polyfit(y[idx], x[idx], 2)
    def x_at(self, fit, y):
        if fit is None:
            return None
        return fit[0]*y*y + fit[1]*y + fit[2]
    
    def mean_x(self,idx, y, x, limit=5):
        if len(idx) == 0:
            return None
        pts = x[idx]

        if len(pts)<limit:
            return None
        return float(np.mean(pts))
    
    def draw_fit(self, fit, color):
        if fit is None:
            return
        
        ploty = np.arange(self.roi_h)

        px = np.clip(
            self.x_at(fit,ploty), 0, self.roi_w-1
        ).astype(int)

        self.vis[ploty,px]=color

    def calc_center(self, left_idx, ly, lx,
                right_idx, ry, rx,
                yellow_idx, yy, yx):

        left_x = self.mean_x(left_idx, ly, lx)
        right_x = self.mean_x(right_idx, ry, rx)
        yellow_x = self.mean_x(yellow_idx, yy, yx)
        
        lane_valid = False
        offset = 0
        lookahead = 0

        y_near = self.roi_h - 1
        y_far = int(self.roi_h * LANE_LOOKAHEAD)

    # 1. 양쪽 차선 모두 검출
        if self.left_fit is not None and self.right_fit is not None:

            left_near = self.x_at(self.left_fit, y_near)
            right_near = self.x_at(self.right_fit, y_near)

            left_far = self.x_at(self.left_fit, y_far)
            right_far = self.x_at(self.right_fit, y_far)

            width = right_near - left_near
            if 180 < width < 400:
                alpha = 0.1
                self.lane_width = (
                   (1 - alpha) * self.lane_width +
                    alpha * width
                )

            near_center = (left_near + right_near) / 2
            far_center = (left_far + right_far) / 2
            lane_valid = True

    # 2. 왼쪽 차선만 검출
        elif self.left_fit is not None:

            near_center = self.x_at(self.left_fit, y_near) + self.lane_width / 2
            far_center = self.x_at(self.left_fit, y_far) + self.lane_width / 2
            lane_valid = True

    # 3. 오른쪽 차선만 검출
        elif self.right_fit is not None:

            near_center = self.x_at(self.right_fit, y_near) - self.lane_width / 2
            far_center = self.x_at(self.right_fit, y_far) - self.lane_width / 2
            lane_valid = True

    # 4. Polyfit 실패 시 평균점 사용
        elif left_x is not None and right_x is not None:

            width = right_x - left_x
            if 180 < width < 400:
                alpha = 0.1
                self.lane_width = (
                    (1 - alpha) * self.lane_width +
                    alpha * width
                )
   
            near_center = (left_x + right_x) / 2
            far_center = near_center
            lane_valid = True

    # 5. 흰 차선을 전혀 못 찾았을 때만 노란 차선 사용
        elif self.yellow_fit is not None:
  
            near_center = self.x_at(self.yellow_fit, y_near)
            far_center = self.x_at(self.yellow_fit, y_far)
            lane_valid = True

        elif yellow_x is not None:

            near_center = yellow_x
            far_center = yellow_x
            lane_valid = True

        if lane_valid:
            offset = near_center - self.roi_w / 2
            lookahead = far_center - self.roi_w / 2

        lane_center = self.roi_w / 2 + offset

        self.visualize(offset)

        return lane_valid, offset, lookahead, lane_center

    def sliding_window(self, mask, base, minpix, color):
        nz = mask.nonzero()
        nzy = np.array(nz[0])
        nzx = np.array(nz[1])

        current = base
        idx = []

        win_h = self.roi_h // SW_NWIN
        
        for win in range(SW_NWIN):
            if current < 0:
                break

            y_low = self.roi_h - (win + 1)*win_h
            y_high = self.roi_h - win*win_h

            if DEBUG_VIZ_LANE:
                cv2.rectangle(
                    self.vis,
                    (current - SW_MARGIN, y_low),
                    (current + SW_MARGIN, y_high),
                    color,
                    2
                )
            good = np.where(
                (nzy >= y_low)&(nzy < y_high)&
                (nzx >= current - SW_MARGIN)&(nzx < current + SW_MARGIN)
            )[0]

            idx.append(good)

            if len(good) > minpix:
                current = int(np.mean(nzx[good]))
        if len(idx):
            idx = np.concatenate(idx)
        else:
            idx = np.array([], dtype=int)

        return idx, nzy, nzx

    def _search_near_fit(self, mask, fit, margin):
        """
        직전 프레임에 유효했던 fit(2차 곡선) 좌우 margin(px) 안에서만 픽셀을 모은다.
        진짜 차선은 프레임 간 위치가 부드럽게 이어지지만, 반사광은 엉뚱한 위치에서
        순간적으로 나타났다 사라지는 경향이 있어 애초에 이 탐색범위 밖이면 무시된다
        (색상/형태와 무관한 시간적 일관성 필터).
          입력 : mask — white/yellow 이진 마스크, fit — 직전 프레임 polyfit 계수, margin — 탐색 반폭(px)
          출력 : sliding_window()와 동일한 형식 (idx, nzy, nzx)
        """
        nz = mask.nonzero()
        nzy = np.array(nz[0])
        nzx = np.array(nz[1])

        if len(nzy) == 0:
            return np.array([], dtype=int), nzy, nzx

        x_pred = self.x_at(fit, nzy)
        idx = np.where(np.abs(nzx - x_pred) < margin)[0]

        if DEBUG_VIZ_LANE:
            ploty = np.arange(self.roi_h)
            x_band = np.clip(self.x_at(fit, ploty), 0, self.roi_w - 1).astype(int)
            for y, x in zip(ploty[::4], x_band[::4]):
                cv2.line(
                    self.vis, (max(x - margin, 0), y), (min(x + margin, self.roi_w - 1), y),
                    (80, 80, 80), 1
                )

        return idx, nzy, nzx

    def _track_lane(self, mask, prev_fit, miss_cnt, base_search, minpix, color, margin=LANE_PRIOR_MARGIN):
        """
        좌/우/노랑 차선 공통 탐색 로직 — prior-based 탐색을 우선 시도하고,
        prev_fit이 없거나 연속 실패가 LANE_PRIOR_MISS_MAX를 넘으면 histogram()+
        sliding_window()로 전체 재탐색(폴백)한다.
          입력 : mask — 이진 마스크, prev_fit — 직전 프레임 fit(없으면 None),
                miss_cnt — 현재까지 연속 실패 횟수, base_search — histogram() 탐색 구간(lo,hi),
                minpix/color — sliding_window()에 그대로 전달
          출력 : (idx, y, x, used_prior) — used_prior는 prior 탐색 사용 여부(디버그용)
        """
        if prev_fit is not None and miss_cnt < LANE_PRIOR_MISS_MAX:
            idx, y, x = self._search_near_fit(mask, prev_fit, margin)
            return idx, y, x, True

        base = self.histogram(mask, base_search)
        idx, y, x = self.sliding_window(mask, base, minpix, color)
        return idx, y, x, False

    def detect(self, bev, white, yellow):
        self.roi_h, self.roi_w = white.shape
        self.vis = bev.copy()

        yellow_minpix = max(SW_MINPIX // 2, 5)

        # ── 좌측 흰 차선 ──
        left_idx, ly, lx, left_prior = self._track_lane(
            white, self.left_fit, self._left_miss_cnt,
            (0, self.roi_w // 2), SW_MINPIX, (0, 255, 0)
        )
        new_left_fit = self.fit_curve(left_idx, ly, lx, SW_MINPIX * 2)
        if new_left_fit is not None:
            self.left_fit = new_left_fit
            self._left_miss_cnt = 0
        else:
            self._left_miss_cnt += 1
            if self._left_miss_cnt >= LANE_PRIOR_MISS_MAX:
                self.left_fit = None   # 오래된 fit 폐기 → 다음 프레임 전체 재탐색 + calc_center 폴백

        # ── 우측 흰 차선 ──
        right_idx, ry, rx, right_prior = self._track_lane(
            white, self.right_fit, self._right_miss_cnt,
            (self.roi_w // 2, self.roi_w), SW_MINPIX, (0, 255, 0)
        )
        new_right_fit = self.fit_curve(right_idx, ry, rx, SW_MINPIX * 2)
        if new_right_fit is not None:
            self.right_fit = new_right_fit
            self._right_miss_cnt = 0
        else:
            self._right_miss_cnt += 1
            if self._right_miss_cnt >= LANE_PRIOR_MISS_MAX:
                self.right_fit = None

        # ── 노란 차선 ──
        yellow_idx, yy, yx, yellow_prior = self._track_lane(
            yellow, self.yellow_fit, self._yellow_miss_cnt,
            (self.roi_w // 4, self.roi_w * 3 // 4), yellow_minpix, (0, 180, 255)
        )
        new_yellow_fit = self.fit_curve(yellow_idx, yy, yx, yellow_minpix * 3)
        if new_yellow_fit is not None:
            self.yellow_fit = new_yellow_fit
            self._yellow_miss_cnt = 0
        else:
            self._yellow_miss_cnt += 1
            if self._yellow_miss_cnt >= LANE_PRIOR_MISS_MAX:
                self.yellow_fit = None

        if DEBUG_VIZ_LANE:
            mode = lambda p: 'PRIOR' if p else 'SCAN'
            cv2.putText(
                self.vis,
                f'L:{mode(left_prior)} R:{mode(right_prior)} Y:{mode(yellow_prior)}',
                (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA
            )

        return self.calc_center(
            left_idx, ly, lx,
            right_idx, ry, rx,
            yellow_idx, yy, yx
            )
    


