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

        #White Lane
        self.white = cv2.inRange(
            hsv, np.array([0,0,180]),np.array([180,45,240])
        )
        #Yellow Lane
        self.yellow = cv2.inRange(
            hsv, np.array([18,120,120]), np.array([35,255,255])
        )
        #Morphology
        kernel = np.ones((3,3), np.uint8)

        self.white = cv2.morphologyEx(
            self.white, cv2.MORPH_OPEN, kernel
        )

        # 세로 성분만 강조
        kernel_vertical = cv2.getStructuringElement(
            cv2.MORPH_RECT, (3,10)
        )

        self.white = cv2.morphologyEx(
            self.white, cv2.MORPH_OPEN, kernel_vertical
        )

        # Connected Components
        num, labels, stats, _ = cv2.connectedComponentsWithStats(self.white)

        mask = np.zeros_like(self.white)

        for i in range(1, num) :
            area = stats[i, cv2.CC_STAT_AREA]

            if 20 < area < 1500:
                mask[labels == i] = 255
            
        self.white = mask

        #yellow morphology
        self.yellow = cv2.morphologyEx(
            self.yellow, cv2.MORPH_CLOSE, kernel
        )

        # Connected Components
        num, labels, stats, _ = cv2.connectedComponentsWithStats(self.yellow)

        mask = np.zeros_like(self.yellow)

        for i in range(1, num):
            area = stats[i, cv2.CC_STAT_AREA]

            width = stats[i, cv2.CC_STAT_WIDTH]

            if 20<area<1000 and width < 40:
                mask[labels == i] = 255

        self.yellow = mask

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
    


