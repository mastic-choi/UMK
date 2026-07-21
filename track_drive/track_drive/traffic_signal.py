import numpy as np
import cv2

#Signal (3구, S0 출발) - Hough Circle 원검출 + 밝기비교
SIG_ROI_T, SIG_ROI_B = 0.17, 0.32
SIG_ROI_L, SIG_ROI_R = 0.32, 0.63
SIG_MIN_RADIUS, SIG_MAX_RADIUS = 12, 28   # 실차 튜닝: 15~25 → 12~28로 완화
SIG_VERT_DIFF_MAX  = SIG_MAX_RADIUS * 2
SIG_HORIZ_DIFF_MAX = SIG_MAX_RADIUS * 8
SIG_MIN_DIST       = SIG_MIN_RADIUS * 3
SIG_BRIGHT_MARGIN  = 15

#Signal4 (4구, S2 교차로) - 배치 좌→우 [빨강,노랑,좌회전,직진]
SIG4_ROI_T, SIG4_ROI_B = 0.08, 0.28
SIG4_ROI_L, SIG4_ROI_R = 0.04, 0.78
SIG4_MIN_RADIUS, SIG4_MAX_RADIUS = 12, 28   # 실차 튜닝: 15~25 → 12~28로 완화
SIG4_VERT_DIFF_MAX  = SIG4_MAX_RADIUS * 2
SIG4_HORIZ_DIFF_MAX = SIG4_MAX_RADIUS * 11
SIG4_MIN_DIST       = SIG4_MIN_RADIUS * 3
SIG4_BRIGHT_MARGIN  = 15

# HoughCircles 엄격도 (실차 튜닝: param1=40→30, param2=20→15로 완화)
#   param1 : Canny 상위임계값(하위임계값은 내부적으로 param1/2) — 낮을수록 흐린 테두리도 엣지로 인정
#   param2 : 원 중심 확정에 필요한 누적표 수 — 낮을수록 불완전한 원도 통과
HOUGH_PARAM1 = 30
HOUGH_PARAM2 = 15

#Debug
DEBUG_VIZ_SIGNAL = True


class SignalDetector:
    def __init__(self):
        self.color = 'unknown'
        self.red_on = False
        self.straight_on = False
        self.left_on = False

        self.roi = None
        self.vis = None

        # ── S0 진단 정보 (CLI 디버그용, track_drive.py의 _print_debug()가 읽어감) ──
        # 원 검출이 실패하는 단계를 구분하기 위한 값들. detect_s0() 호출마다 갱신됨.
        self.s0_roi_px      = (0, 0, 0, 0)   # (t, b, l, r) — ROI 픽셀 좌표(원본 프레임 기준)
        self.s0_circle_count = 0             # HoughCircles가 찾은 원 개수(0=아예 못 찾음)
        self.s0_reject_reason = 'no_frame'   # 실패 사유 문자열, 성공 시 None
        self.s0_brightness   = []            # 3개 원의 밝기값(성공적으로 3개+배치통과 시에만 채워짐)
        self.s0_bright_margin = 0.0          # 최댓값-평균 (SIG_BRIGHT_MARGIN과 비교되는 실측값)

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
        # 반환값: (통과여부, 실패시 사유 문자열 — CLI 디버그 로그용, 통과 시 None)
        xs = sorted(int(c[0]) for c in circles)
        ys = sorted(int(c[1]) for c in circles)

        vert_spread = ys[-1] - ys[0]
        if vert_spread > vert_max:
            return False, f'vert_spread={vert_spread}>{vert_max}'

        horiz_spread = xs[-1] - xs[0]
        if horiz_spread > horiz_max:
            return False, f'horiz_spread={horiz_spread}>{horiz_max}'

        for i in range(len(xs) - 1):
            gap = xs[i + 1] - xs[i]
            if gap < min_dist:
                return False, f'gap[{i}]={gap}<{min_dist}'
        return True, None

    def find_circles(self, roi, min_r, max_r, tag='sig'):
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        circles = cv2.HoughCircles(
            blur, cv2.HOUGH_GRADIENT, 1, 20,
            param1=HOUGH_PARAM1, param2=HOUGH_PARAM2,
            minRadius=min_r, maxRadius=max_r
        )

        if DEBUG_VIZ_SIGNAL:
            # HoughCircles가 내부적으로 보는 엣지맵과 동일한 임계값(param1, param1/2)으로 재현.
            # 원이 안 잡힐 때 "엣지 자체가 안 그려지는지" vs "엣지는 있는데 원으로 안 뭉치는지"
            # 구분하기 위한 디버그 창. tag로 s0/s2 창을 구분한다.
            edges = cv2.Canny(blur, HOUGH_PARAM1 // 2, HOUGH_PARAM1)
            cv2.imshow(f'canny_{tag}', edges)
            cv2.waitKey(1)

        return gray, circles

    def detect_s0(self, frame):
        if frame is None:
            self.s0_reject_reason = 'no_frame'
            return self.color

        h, w = frame.shape[:2]
        t, b = int(h*SIG_ROI_T), int(h*SIG_ROI_B)
        l, r = int(w*SIG_ROI_L), int(w*SIG_ROI_R)
        self.s0_roi_px = (t, b, l, r)

        self.roi = frame[t:b, l:r]

        gray, circles = self.find_circles(self.roi, SIG_MIN_RADIUS, SIG_MAX_RADIUS, tag='s0')
        self.color = 'unknown'
        self.s0_circle_count = 0
        self.s0_reject_reason = 'no_circles_found'
        self.s0_brightness = []
        self.s0_bright_margin = 0.0

        if circles is not None:
            circles = np.round(circles[0, :]).astype(int)
            self.s0_circle_count = len(circles)

            if len(circles) != 3:
                self.s0_reject_reason = f'circle_count={len(circles)}(need 3)'
            else:
                shape_pass, reason = self.shape_ok(circles, SIG_VERT_DIFF_MAX, SIG_HORIZ_DIFF_MAX, SIG_MIN_DIST)
                if not shape_pass:
                    self.s0_reject_reason = reason
                else:
                    circles = sorted(circles, key=lambda c: c[0])   #좌→우: 빨강,노랑,초록
                    bright = [self.circle_brightness(gray, x, y, r) for x, y, r in circles]
                    self.s0_brightness = [round(b_, 1) for b_ in bright]

                    idx = int(np.argmax(bright))
                    margin = bright[idx] - float(np.mean(bright))
                    self.s0_bright_margin = round(margin, 1)
                    if margin > SIG_BRIGHT_MARGIN:
                        self.color = ('red', 'yellow', 'blue')[idx]   #idx=2(우측)=초록=출발
                        self.s0_reject_reason = None
                    else:
                        self.s0_reject_reason = f'bright_margin={margin:.1f}<={SIG_BRIGHT_MARGIN}'

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

        gray, circles = self.find_circles(self.roi, SIG4_MIN_RADIUS, SIG4_MAX_RADIUS, tag='s2')
        self.red_on = self.straight_on = self.left_on = False

        if circles is not None:
            circles = np.round(circles[0, :]).astype(int)

            shape_pass, _reason = self.shape_ok(circles, SIG4_VERT_DIFF_MAX, SIG4_HORIZ_DIFF_MAX, SIG4_MIN_DIST)
            if len(circles) == 4 and shape_pass:
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
