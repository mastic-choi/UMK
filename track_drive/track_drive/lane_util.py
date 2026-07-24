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
LANE_ROI_TOP = 0.5125
LANE_ROI_BOT = 0.825
LANE_LOOKAHEAD = 0.35  # KUAC_2024엔 예측조향 개념 자체가 없어 대응값 없음 — 기존값 유지
# Yellow Lane
LANE_YELLOW_WEIGHT = 0.25
LANE_YELLOW_MAX_DEV = 40

# --- Adaptive Thresholding 파라미터 ---
# White: adaptiveThreshold(V채널) 사용. blockSize는 홀수, ROI 폭(대략 320~640px)에 맞춰 튜닝.
WHITE_ADAPT_BLOCK = 25      # 지역 윈도우 크기 (홀수)
WHITE_ADAPT_C = -15         # 평균보다 얼마나 밝아야 흰색으로 볼지 (음수 = 임계값을 낮춰 더 잘 잡음)
WHITE_S_MAX = 45            # 채도 상한은 기존 inRange 값 유지 (흰색=저채도 필터링)
WHITE_MIN_V_FLOOR = 120     # 너무 어두운 프레임에서 노이즈가 전부 잡히는 것 방지용 하한

# Yellow: 색상(H)은 고정, S/V 하한만 프레임 밝기 percentile로 동적 조정
YELLOW_H_RANGE = (15, 40)
YELLOW_S_PERCENTILE = 70    # 노란색 후보 영역의 채도 분포 기준 (동적 하한 산출용)
YELLOW_S_MIN_FLOOR = 60
YELLOW_V_PERCENTILE = 70
YELLOW_V_MIN_FLOOR = 60

#Debug
DEBUG_VIZ_LANE = True

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

    def _adaptive_white_mask(self, hsv):
        v = hsv[:, :, 2]
        s = hsv[:, :, 1]

        # blockSize는 반드시 홀수 & 3 이상이어야 함 (ROI가 작을 경우 자동 보정)
        block = WHITE_ADAPT_BLOCK
        block = min(block, max(3, (min(self.roi_h, self.roi_w) // 2) | 1))
        if block % 2 == 0:
            block += 1

        adapt = cv2.adaptiveThreshold(
            v, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            block,
            WHITE_ADAPT_C
        )

        # 절대적으로 너무 어두운 픽셀(그림자 내부 등)은 지역 대비만으로 잡혔더라도 제외
        floor_mask = cv2.inRange(v, WHITE_MIN_V_FLOOR, 255)
        sat_mask = cv2.inRange(s, 0, WHITE_S_MAX)

        return cv2.bitwise_and(cv2.bitwise_and(adapt, floor_mask), sat_mask)

    def _adaptive_yellow_mask(self, hsv):
        h, s, v = cv2.split(hsv)

        # H 범위 내 후보 픽셀만으로 S/V 분포를 계산해야 아스팔트 등 배경에 안 휘둘림
        h_mask = cv2.inRange(h, YELLOW_H_RANGE[0], YELLOW_H_RANGE[1])
        candidate = h_mask > 0

        if np.count_nonzero(candidate) > 0:
            s_thresh = max(YELLOW_S_MIN_FLOOR, np.percentile(s[candidate], YELLOW_S_PERCENTILE) - 20)
            v_thresh = max(YELLOW_V_MIN_FLOOR, np.percentile(v[candidate], YELLOW_V_PERCENTILE) - 20)
        else:
            s_thresh, v_thresh = YELLOW_S_MIN_FLOOR, YELLOW_V_MIN_FLOOR

        return cv2.inRange(
            hsv,
            np.array([YELLOW_H_RANGE[0], s_thresh, v_thresh]),
            np.array([YELLOW_H_RANGE[1], 255, 255])
        )

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

        #HSV
        hsv = cv2.cvtColor(self.bev, cv2.COLOR_BGR2HSV)

        #White Lane (adaptiveThreshold 기반)
        self.white = self._adaptive_white_mask(hsv)

        #Yellow Lane (동적 S/V 하한 기반)
        self.yellow = self._adaptive_yellow_mask(hsv)

        #Morphology
        kernel = np.ones((3,3), np.uint8)

        self.white = cv2.morphologyEx(
            self.white, cv2.MORPH_OPEN, kernel
        )

        self.yellow = cv2.morphologyEx(
            self.yellow, cv2.MORPH_CLOSE, kernel
        )

        #DEBUG
        if DEBUG_VIZ_LANE:
            cv2.imshow("lane_bev", self.bev)
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
        y_far = int(self.roi_h*LANE_LOOKAHEAD)

        if self.left_fit is not None and self.right_fit is not None:
            near_center = (
                self.x_at(self.left_fit, y_near)+
                self.x_at(self.right_fit, y_near)
            ) / 2

            far_center = (
                self.x_at(self.left_fit, y_far)+
                self.x_at(self.right_fit, y_far)
            ) / 2

            offset = near_center - self.roi_w/2
            lookahead = far_center - self.roi_w/2

            lane_valid = True

        elif left_x is not None and right_x is not None:
            center = (left_x + right_x) / 2
            offset = center - self.roi_w/2
            lookahead = offset
            lane_valid = True
        
        elif yellow_x is not None:
            offset = yellow_x - self.roi_w/2
            lookahead = offset
            lane_valid = True
        
        # 차선 중앙 x좌표(px) — offset 정의(center - roi_w/2)를 역산해 세 분기 모두에서 일관되게 산출.
        # 미검출 시 offset=0 → 화면 중앙(roi_w/2)이 기본값이 된다.
        # track_drive.perc_obstacle_lane()에서 YOLO bbox 중심과 비교해 장애물 좌/우 판단에 사용.
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
    
    def detect(self, bev, white, yellow):
        self.roi_h, self.roi_w = white.shape
        self.vis = bev.copy()

        #histogram
        self.left_base = self.histogram(
            white,
            (0, self.roi_w // 2)
        )

        self.right_base = self.histogram(
            white,
            (self.roi_w // 2, self.roi_w)
        )

        yellow_base = self.histogram(
            yellow,
            (self.roi_w // 4, self.roi_w*3 // 4)
        )

        #sliding window
        left_idx, ly, lx = self.sliding_window(
            white, self.left_base, SW_MINPIX, (0,255,0)
        )

        right_idx, ry, rx = self.sliding_window(
            white, self.right_base, SW_MINPIX, (0,255,0)
        )

        yellow_idx, yy, yx = self.sliding_window(
            yellow, yellow_base, max(SW_MINPIX//2,5),(0,180,255)
        )

        #polyfit
        self.left_fit = self.fit_curve(
            left_idx, ly, lx, SW_MINPIX*2
        )

        self.right_fit = self.fit_curve(
            right_idx, ry, rx, SW_MINPIX*2
        )

        self.yellow_fit = self.fit_curve(
            yellow_idx, yy, yx, max(SW_MINPIX//2, 5)*3
        )

        return self.calc_center(
            left_idx, ly, lx,
            right_idx, ry, rx,
            yellow_idx, yy, yx
            )