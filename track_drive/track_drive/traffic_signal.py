import numpy as np
import cv2

#Signal (3구, S0 출발) - Hough Circle 원검출 + 밝기비교
SIG_ROI_T, SIG_ROI_B = 0.17, 0.32
SIG_ROI_L, SIG_ROI_R = 0.32, 0.63
SIG_MIN_RADIUS, SIG_MAX_RADIUS = 15, 25
SIG_VERT_DIFF_MAX  = SIG_MAX_RADIUS * 2
SIG_HORIZ_DIFF_MAX = SIG_MAX_RADIUS * 8
SIG_MIN_DIST       = SIG_MIN_RADIUS * 3
SIG_BRIGHT_MARGIN  = 15

#Signal4 (4구, S2 교차로) - 배치 좌→우 [빨강,노랑,좌회전,직진]
SIG4_ROI_T, SIG4_ROI_B = 0.08, 0.28
SIG4_ROI_L, SIG4_ROI_R = 0.04, 0.78
SIG4_MIN_RADIUS, SIG4_MAX_RADIUS = 15, 25
SIG4_VERT_DIFF_MAX  = SIG4_MAX_RADIUS * 2
SIG4_HORIZ_DIFF_MAX = SIG4_MAX_RADIUS * 11
SIG4_MIN_DIST       = SIG4_MIN_RADIUS * 3
SIG4_BRIGHT_MARGIN  = 15

#Debug
DEBUG_VIZ_SIGNAL = False


class SignalDetector:
    def __init__(self):
        self.color = 'unknown'
        self.red_on = False
        self.straight_on = False
        self.left_on = False

        self.roi = None
        self.vis = None

    def circle_brightness(self, gray, x, y, r):
        y0, y1 = max(0, y - r // 2), y + r // 2
        x0, x1 = max(0, x - r // 2), x + r // 2
        patch = gray[y0:y1, x0:x1]

        if patch.size == 0:
            return 0.0
        return float(np.mean(patch))

    def shape_ok(self, circles, vert_max, horiz_max, min_dist):
        # ★TODO(실차 테스트시 체크): 원 개수가 정확히 3개/4개가 아니면(빛반사·블러로 하나 덜/더 잡히면)
        #   그 프레임은 무조건 인식 실패로 처리됨 — 디바운스나 "N프레임 중 M번" 같은 폴백이 없음.
        #   실차 카메라로 돌려보고 신호 놓치는 빈도가 높으면 여기에 보강 필요.
        xs = sorted(int(c[0]) for c in circles)
        ys = sorted(int(c[1]) for c in circles)

        if (ys[-1] - ys[0]) > vert_max:
            return False
        if (xs[-1] - xs[0]) > horiz_max:
            return False

        for i in range(len(xs) - 1):
            if (xs[i + 1] - xs[i]) < min_dist:
                return False
        return True

    def find_circles(self, roi, min_r, max_r):
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        circles = cv2.HoughCircles(
            blur, cv2.HOUGH_GRADIENT, 1, 20,
            param1=40, param2=20,
            minRadius=min_r, maxRadius=max_r
        )

        return gray, circles

    def detect_s0(self, frame):
        if frame is None:
            return self.color

        h, w = frame.shape[:2]

        self.roi = frame[
            int(h*SIG_ROI_T): int(h*SIG_ROI_B),
            int(w*SIG_ROI_L): int(w*SIG_ROI_R)
            ]

        gray, circles = self.find_circles(self.roi, SIG_MIN_RADIUS, SIG_MAX_RADIUS)
        self.color = 'unknown'

        if circles is not None:
            circles = np.round(circles[0, :]).astype(int)

            if len(circles) == 3 and self.shape_ok(circles, SIG_VERT_DIFF_MAX, SIG_HORIZ_DIFF_MAX, SIG_MIN_DIST):
                circles = sorted(circles, key=lambda c: c[0])   #좌→우: 빨강,노랑,초록
                bright = [self.circle_brightness(gray, x, y, r) for x, y, r in circles]

                idx = int(np.argmax(bright))
                if bright[idx] - float(np.mean(bright)) > SIG_BRIGHT_MARGIN:
                    self.color = ('red', 'yellow', 'blue')[idx]   #idx=2(우측)=초록=출발

        if DEBUG_VIZ_SIGNAL:
            self.vis = self.roi.copy()
            cv2.putText(
                self.vis, self.color, (4, 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA
            )
            cv2.imshow('signal_roi', self.vis)
            cv2.waitKey(1)

        return self.color

    def detect_s2(self, frame):
        if frame is None:
            return self.red_on, self.straight_on, self.left_on

        h, w = frame.shape[:2]

        self.roi = frame[
            int(h*SIG4_ROI_T): int(h*SIG4_ROI_B),
            int(w*SIG4_ROI_L): int(w*SIG4_ROI_R)
            ]

        gray, circles = self.find_circles(self.roi, SIG4_MIN_RADIUS, SIG4_MAX_RADIUS)
        self.red_on = self.straight_on = self.left_on = False

        if circles is not None:
            circles = np.round(circles[0, :]).astype(int)

            if len(circles) == 4 and self.shape_ok(circles, SIG4_VERT_DIFF_MAX, SIG4_HORIZ_DIFF_MAX, SIG4_MIN_DIST):
                circles = sorted(circles, key=lambda c: c[0])   #좌→우: 빨강,노랑,좌회전,직진
                bright = [self.circle_brightness(gray, x, y, r) for x, y, r in circles]
                avg = float(np.mean(bright))

                lit = [b - avg > SIG4_BRIGHT_MARGIN for b in bright]
                red_lit, _yellow_lit, left_lit, straight_lit = lit

                self.left_on     = left_lit
                self.straight_on = straight_lit and not left_lit
                self.red_on      = red_lit and not (left_lit or straight_lit)

        if DEBUG_VIZ_SIGNAL:
            self.vis = self.roi.copy()
            state = ('LEFT' if self.left_on else
                     'STR'  if self.straight_on else
                     'RED'  if self.red_on else '---')
            color = ((0, 255, 0) if state in ('LEFT', 'STR') else
                     (0, 0, 255) if state == 'RED' else (180, 180, 180))
            cv2.putText(self.vis, state, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
            cv2.imshow('signal4_roi', self.vis)
            cv2.waitKey(1)

        return self.red_on, self.straight_on, self.left_on
