import itertools

import numpy as np
import cv2

# Signal4 (4구) - 배치 좌→우 [빨강,노랑,좌회전,직진]
#   대회 규정 변경(2026-07): 출발(S0)도 교차로(S2)와 동일한 4구 신호등을 재사용한다.
#     S0: 초록(직진 위치)만 점등 → 출발
#     S2: 초록만 점등 → 직진 / 초록+빨강 동시 점등 → 좌회전
#   S0/S2 물리적 카메라-신호등 거리가 같아 ROI/반지름도 공유한다(실측 확인됨).
SIG4_ROI_T, SIG4_ROI_B = 0.08, 0.28
SIG4_ROI_L, SIG4_ROI_R = 0.04, 0.78
SIG4_MIN_RADIUS, SIG4_MAX_RADIUS = 15, 25
SIG4_VERT_DIFF_MAX  = SIG4_MAX_RADIUS * 2
SIG4_HORIZ_DIFF_MAX = SIG4_MAX_RADIUS * 11
SIG4_MIN_DIST       = SIG4_MIN_RADIUS * 3
SIG4_BRIGHT_MARGIN  = 15
SIG4_MAX_CANDIDATES = 10  # 원이 이보다 많이 잡히면 조합 탐색 없이 바로 실패 처리(ROI 자체가 노이즈로 판단)

#Debug
DEBUG_VIZ_SIGNAL = False


class SignalDetector:
    def __init__(self):
        self.red_on = False
        self.straight_on = False
        self.left_on = False

        self.roi = None
        self.vis = None

        # ── [진단] track_drive._print_debug() 가 S0/S2 상태에서 읽는 값들 ──
        #   신호등 인식이 '어느 단계에서' 막혔는지 로그로 좁히기 위한 것.
        #   S0도 S2와 동일한 detect_s2()를 재사용하므로(대회 규정 변경) 진단 필드도 하나로 통합.
        #   이 속성들이 없으면 START_STATE 를 S0_WAIT_GREEN 으로 되돌리는 순간
        #   _print_debug() 가 AttributeError 로 죽는다.
        self.s2_roi_px        = (0, 0, 0, 0)  # ROI 픽셀좌표 (top, bottom, left, right)
        self.s2_circle_count  = 0             # HoughCircles 가 찾은 원 개수
        self.s2_reject_reason = ''            # 실패 사유 ('' = 성공)
        self.s2_brightness    = []            # 좌→우 (빨강,노랑,좌회전,직진) 밝기

    def circle_brightness(self, gray, x, y, r):
        y0, y1 = max(0, y - r // 2), y + r // 2
        x0, x1 = max(0, x - r // 2), x + r // 2
        patch = gray[y0:y1, x0:x1]

        if patch.size == 0:
            return 0.0
        return float(np.mean(patch))

    def shape_ok(self, circles, vert_max, horiz_max, min_dist):
        """배치 검사(원 정확히 4개 대상). 반환 (통과여부, 실패사유) — 사유는 디버그 로그용, 통과 시 ''.
        4개보다 많이 잡힌 경우의 완화 처리는 pick_best_4()가 담당. 4개 미만(가림/블러로 실제
        누락)은 이 함수까지 안 오고 detect_s2()에서 바로 실패 처리됨 — 없는 원을 만들어낼 근거가
        없기 때문. 프레임 단위 디바운스는 track_drive.py의 perc_signal()에서 처리."""
        xs = sorted(int(c[0]) for c in circles)
        ys = sorted(int(c[1]) for c in circles)

        if (ys[-1] - ys[0]) > vert_max:
            return False, f'vert_spread={ys[-1] - ys[0]}>{vert_max}'
        if (xs[-1] - xs[0]) > horiz_max:
            return False, f'horiz_spread={xs[-1] - xs[0]}>{horiz_max}'

        for i in range(len(xs) - 1):
            if (xs[i + 1] - xs[i]) < min_dist:
                return False, f'gap[{i}]={xs[i + 1] - xs[i]}<{min_dist}'
        return True, ''

    def pick_best_4(self, circles, vert_max, horiz_max, min_dist):
        """원이 4개보다 많이 잡혔을 때(반사광 등 오검출 섞임), 신호등 배치(shape_ok)를
        통과하는 4개 조합 중 가장 그럴듯한 것을 고른다.
        여러 조합이 통과하면 세로 퍼짐(vert_spread)이 가장 작은 걸 택한다 — 같은 높이에
        나란히 있을수록 진짜 신호등일 확률이 높다는 가정. 반환 (선택된 4개 또는 None, 실패사유)."""
        best, best_score = None, None
        for combo in itertools.combinations(circles, 4):
            ok, _ = self.shape_ok(combo, vert_max, horiz_max, min_dist)
            if not ok:
                continue
            ys = [c[1] for c in combo]
            score = max(ys) - min(ys)
            if best_score is None or score < best_score:
                best, best_score = combo, score

        if best is None:
            return None, 'no_valid_4subset'
        return list(best), ''

    def find_circles(self, roi, min_r, max_r):
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        circles = cv2.HoughCircles(
            blur, cv2.HOUGH_GRADIENT, 1, 20,
            param1=40, param2=20,
            minRadius=min_r, maxRadius=max_r
        )

        return gray, circles

    def detect_s2(self, frame):
        """4구 신호등 인식 — S0(출발)/S2(교차로) 공통.
          S0에서 쓸 때: straight_on(초록만 점등) = 출발
          S2에서 쓸 때: straight_on = 직진, left_on(초록+빨강 동시) = 좌회전

        원이 정확히 4개가 아니어도 어느 정도는 견딘다(반사광 등으로 여분의 원이 섞이는 경우가
        실차에서 흔해서): 4개 초과면 pick_best_4()로 가장 신호등다운 4개 조합을 고른다.
        4개 미만이면(가림/블러로 실제로 못 찾은 경우) 만들어낼 근거가 없으므로 그대로 실패.
        프레임 단위 디바운스(연속 N프레임 확정)는 track_drive.py의 perc_signal()에서 처리한다
        (여기는 항상 '이번 한 프레임' 기준 순간값만 반환)."""
        if frame is None:
            return self.red_on, self.straight_on, self.left_on

        h, w = frame.shape[:2]

        t, b = int(h*SIG4_ROI_T), int(h*SIG4_ROI_B)
        l, r_ = int(w*SIG4_ROI_L), int(w*SIG4_ROI_R)
        self.roi = frame[t:b, l:r_]

        gray, circles = self.find_circles(self.roi, SIG4_MIN_RADIUS, SIG4_MAX_RADIUS)
        self.red_on = self.straight_on = self.left_on = False

        # 진단값 초기화 (이번 프레임 기준)
        self.s2_roi_px        = (t, b, l, r_)
        self.s2_circle_count  = 0 if circles is None else len(circles[0])
        self.s2_reject_reason = 'no_circles'
        self.s2_brightness    = []

        if circles is not None:
            circles = np.round(circles[0, :]).astype(int)
            n = len(circles)

            if n < 4:
                self.s2_reject_reason = f'circle_count={n}(<4)'
            elif n > SIG4_MAX_CANDIDATES:
                self.s2_reject_reason = f'circle_count={n}(>{SIG4_MAX_CANDIDATES}, too noisy)'
            else:
                if n == 4:
                    ok, reason = self.shape_ok(circles, SIG4_VERT_DIFF_MAX, SIG4_HORIZ_DIFF_MAX, SIG4_MIN_DIST)
                    chosen = list(circles) if ok else None
                else:
                    chosen, reason = self.pick_best_4(circles, SIG4_VERT_DIFF_MAX, SIG4_HORIZ_DIFF_MAX, SIG4_MIN_DIST)

                if chosen is None:
                    self.s2_reject_reason = f'circle_count={n} {reason}'
                else:
                    self.s2_reject_reason = ''
                    circles_sorted = sorted(chosen, key=lambda c: c[0])   #좌→우: 빨강,노랑,좌회전,직진
                    bright = [self.circle_brightness(gray, x, y, r) for x, y, r in circles_sorted]
                    avg = float(np.mean(bright))
                    self.s2_brightness = [round(v, 1) for v in bright]

                    lit = [bv - avg > SIG4_BRIGHT_MARGIN for bv in bright]
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
