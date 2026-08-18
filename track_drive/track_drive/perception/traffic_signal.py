import itertools

import numpy as np
import cv2

# Signal4 (4구) - 배치 좌→우 [빨강,노랑,좌회전,직진]
#   대회 규정 변경(2026-07): 출발(S0)도 교차로(S2)와 동일한 4구 신호등을 재사용한다.
#     S0: 초록(직진 위치)만 점등 → 출발
#     S2: 초록만 점등 → 직진 / 초록+빨강 동시 점등 → 좌회전
#   S0/S2 물리적 카메라-신호등 거리가 같아 ROI/반지름도 공유한다(실측 확인됨).
#   기본 튜닝값(SIG4_ROI_*/MIN_RADIUS/MAX_RADIUS/BRIGHT_MARGIN/MAX_CANDIDATES,
#   DEBUG_VIZ_SIGNAL)은 config.py에 있다 — 실차 테스트 중 값을 바꾸려면 이 파일이
#   아니라 config.py를 고칠 것. 아래 셋(VERT/HORIZ_DIFF_MAX, MIN_DIST)은
#   MIN/MAX_RADIUS에서 파생되는 값이라 "N배" 관계를 그대로 보여주려고 여기 둔다.
from ..config import (
    SIG4_ROI_T, SIG4_ROI_B, SIG4_ROI_L, SIG4_ROI_R,
    SIG4_MIN_RADIUS, SIG4_MAX_RADIUS, SIG4_BRIGHT_MARGIN, SIG4_MAX_CANDIDATES,
    SIG4_AUTOCROP_ENABLE, SIG4_SEARCH_ROI_T, SIG4_SEARCH_ROI_B,
    SIG4_SEARCH_ROI_L, SIG4_SEARCH_ROI_R, SIG4_BOARD_GRAY_MIN,
    SIG4_BOARD_MIN_AREA_PX, SIG4_BOARD_MAX_AREA_PX, SIG4_BOARD_MIN_ASPECT,
    SIG4_BOARD_MAX_ASPECT, SIG4_BOARD_MAX_CANDIDATES, SIG4_BOARD_CROP_MARGIN,
    DEBUG_VIZ_SIGNAL,
)
SIG4_VERT_DIFF_MAX  = SIG4_MAX_RADIUS * 2
SIG4_HORIZ_DIFF_MAX = SIG4_MAX_RADIUS * 11
SIG4_MIN_DIST       = SIG4_MIN_RADIUS * 3

# [2026-08-13] 디버그 창 전용 확대 배율. ROI가 원본 프레임의 일부(SIG4_ROI_*)라 원본
# 그대로 띄우면 원/글자가 너무 작아 확인이 힘들다 — 판정 로직과 무관한 순수 표시값이라
# config.py가 아니라 여기 상수로 둔다(튜닝 대상 아님).
SIG4_VIZ_SCALE = 3
# 좌→우 배치 라벨. 한글(빨강/노랑/좌회전/직진)은 OpenCV 기본 폰트(Hershey)가 한글 글리프를
# 지원하지 않아 이미지 위에는 깨져 나온다 — 이미지 오버레이는 영문 약어, 한글 설명은 터미널
# 로그(track_drive.py DEBUG_LOG_SIGNAL) 쪽에서 담당한다.
SIG4_LABELS = ('R', 'Y', 'L', 'S')  # 빨강, 노랑, 좌회전, 직진


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
        self.s2_lit           = []            # 좌→우 점등 여부(밝기가 평균보다 SIG4_BRIGHT_MARGIN 이상) — 배치검사 통과 시에만 채워짐

        # ── [2026-08-18 신규, 실차 미검증] 흰 배경판 자동크롭 후보 진단 ──
        #   detect_s2()가 이번 프레임에 시도한 ROI 후보 전부(자동탐색 + 마지막 고정폴백)와
        #   그중 실제로 성공한 인덱스. _draw_debug_board_search()가 이걸로 signal4_board_search
        #   창을 그린다 — 어떤 후보가 왜 채택/기각됐는지 실차에서 눈으로 확인하기 위함.
        self.s2_board_candidates = []   # [(t,b,l,r), ...] — 마지막 항목이 항상 고정폴백(SIG4_ROI_*)
        self.s2_chosen_idx       = -1   # 성공한 후보의 인덱스, -1이면 전부 실패

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

    def _board_candidates(self, frame):
        """흰 배경판(신호등 rig) 후보 사각형을 탐색범위(SIG4_SEARCH_ROI_*) 안에서 찾아
        (t,b,l,r) 절대 픽셀좌표 리스트로 반환한다(마진 포함, 프레임 경계로 클리핑됨,
        면적 큰 순 정렬). 재현율 우선 설계 — 천장 조명 등 오탐이 섞여 나와도 상관없다는
        전제다: 이 함수는 "후보 제안"만 하고, 그중 진짜 신호등인지는 detect_s2()의 기존
        원 4개 패턴검사(find_circles/shape_ok/pick_best_4)가 최종 판정한다. 실패해도
        예외 없이 빈 리스트만 반환 — 호출부가 SIG4_ROI_* 고정 폴백으로 이어가면 된다."""
        h, w = frame.shape[:2]
        st, sb = int(h * SIG4_SEARCH_ROI_T), int(h * SIG4_SEARCH_ROI_B)
        sl, sr = int(w * SIG4_SEARCH_ROI_L), int(w * SIG4_SEARCH_ROI_R)
        band = frame[st:sb, sl:sr]
        if band.size == 0:
            return []

        gray = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, SIG4_BOARD_GRAY_MIN, 255, cv2.THRESH_BINARY)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        scored = []
        for c in contours:
            area = cv2.contourArea(c)
            if not (SIG4_BOARD_MIN_AREA_PX <= area <= SIG4_BOARD_MAX_AREA_PX):
                continue
            x, y, cw, ch = cv2.boundingRect(c)
            if ch == 0:
                continue
            aspect = cw / ch
            if not (SIG4_BOARD_MIN_ASPECT <= aspect <= SIG4_BOARD_MAX_ASPECT):
                continue

            mx, my = int(cw * SIG4_BOARD_CROP_MARGIN), int(ch * SIG4_BOARD_CROP_MARGIN)
            # band-로컬 좌표 → 원본 frame 절대좌표 변환 + 마진 + 프레임 경계 클리핑
            l  = max(0, sl + x - mx)
            r_ = min(w, sl + x + cw + mx)
            t  = max(0, st + y - my)
            b  = min(h, st + y + ch + my)
            scored.append((area, t, b, l, r_))

        scored.sort(key=lambda v: v[0], reverse=True)
        return [(t, b, l, r_) for _, t, b, l, r_ in scored[:SIG4_BOARD_MAX_CANDIDATES]]

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

        [2026-08-18] ROI를 SIG4_ROI_*(고정 크롭) 하나가 아니라 "후보 리스트"로 순회한다 —
        SIG4_AUTOCROP_ENABLE=True면 _board_candidates()가 흰 배경판(신호등 rig) 후보를 먼저
        제안하고, 그 뒤에 항상 기존 SIG4_ROI_*(고정 크롭)를 마지막 안전망으로 덧붙인다. 각
        후보에 대해 아래 원 4개 패턴검사(find_circles→shape_ok/pick_best_4, 기존 로직 그대로)를
        순서대로 시도해 맨 처음 성공하는 후보를 채택하고 즉시 멈춘다 — 자동탐색이 전부
        실패해도 도입 전과 똑같이 고정 크롭으로 마지막에 한 번 더 시도되므로 이 변경으로
        인식률이 이전보다 나빠지진 않는다.

        원이 정확히 4개가 아니어도 어느 정도는 견딘다(반사광 등으로 여분의 원이 섞이는 경우가
        실차에서 흔해서): 4개 초과면 pick_best_4()로 가장 신호등다운 4개 조합을 고른다.
        4개 미만이면(가림/블러로 실제로 못 찾은 경우) 만들어낼 근거가 없으므로 그 후보는 실패
        처리하고 다음 후보로 넘어간다.
        프레임 단위 디바운스(연속 N프레임 확정)는 track_drive.py의 perc_signal()에서 처리한다
        (여기는 항상 '이번 한 프레임' 기준 순간값만 반환)."""
        if frame is None:
            return self.red_on, self.straight_on, self.left_on

        h, w = frame.shape[:2]

        roi_boxes = list(self._board_candidates(frame)) if SIG4_AUTOCROP_ENABLE else []
        fixed_t, fixed_b = int(h * SIG4_ROI_T), int(h * SIG4_ROI_B)
        fixed_l, fixed_r = int(w * SIG4_ROI_L), int(w * SIG4_ROI_R)
        roi_boxes.append((fixed_t, fixed_b, fixed_l, fixed_r))  # 항상 마지막 안전망

        self.s2_board_candidates = roi_boxes
        self.s2_chosen_idx = -1

        self.red_on = self.straight_on = self.left_on = False
        all_circles = chosen = gray = None

        for idx, (t, b, l, r_) in enumerate(roi_boxes):
            roi = frame[t:b, l:r_]
            gray, raw_circles = self.find_circles(roi, SIG4_MIN_RADIUS, SIG4_MAX_RADIUS)

            # 진단값(이번 후보 기준으로 매번 덮어씀 — 루프가 끝났을 때 마지막으로 시도한
            # 후보의 값이 남는다. 성공하면 break하므로 그게 곧 채택된 후보의 값)
            self.roi = roi
            self.s2_roi_px        = (t, b, l, r_)
            self.s2_circle_count  = 0 if raw_circles is None else len(raw_circles[0])
            self.s2_reject_reason = 'no_circles'
            self.s2_brightness    = []
            self.s2_lit           = []
            all_circles = None
            chosen      = None

            if raw_circles is not None:
                all_circles = np.round(raw_circles[0, :]).astype(int)
                n = len(all_circles)

                if n < 4:
                    self.s2_reject_reason = f'circle_count={n}(<4)'
                elif n > SIG4_MAX_CANDIDATES:
                    self.s2_reject_reason = f'circle_count={n}(>{SIG4_MAX_CANDIDATES}, too noisy)'
                else:
                    if n == 4:
                        ok, reason = self.shape_ok(all_circles, SIG4_VERT_DIFF_MAX, SIG4_HORIZ_DIFF_MAX, SIG4_MIN_DIST)
                        picked = list(all_circles) if ok else None
                    else:
                        picked, reason = self.pick_best_4(all_circles, SIG4_VERT_DIFF_MAX, SIG4_HORIZ_DIFF_MAX, SIG4_MIN_DIST)

                    if picked is None:
                        self.s2_reject_reason = f'circle_count={n} {reason}'
                    else:
                        self.s2_reject_reason = ''
                        chosen = sorted(picked, key=lambda c: c[0])   #좌→우: 빨강,노랑,좌회전,직진

            if chosen is not None:
                self.s2_chosen_idx = idx
                break

        if chosen is not None:
            bright = [self.circle_brightness(gray, x, y, r) for x, y, r in chosen]
            avg = float(np.mean(bright))
            self.s2_brightness = [round(v, 1) for v in bright]

            lit = [bv - avg > SIG4_BRIGHT_MARGIN for bv in bright]
            self.s2_lit = lit
            red_lit, _yellow_lit, left_lit, straight_lit = lit

            self.left_on     = left_lit
            self.straight_on = straight_lit and not left_lit
            self.red_on      = red_lit and not (left_lit or straight_lit)

        if DEBUG_VIZ_SIGNAL:
            self._draw_debug_viz(all_circles, chosen)
            self._draw_debug_board_search(frame, roi_boxes, self.s2_chosen_idx)

        return self.red_on, self.straight_on, self.left_on

    def _draw_debug_viz(self, all_circles, chosen):
        """'signal4_roi' 창: ROI 크롭을 SIG4_VIZ_SCALE배 확대해 그 위에
          - 노란 원: Hough가 찾은 원 전부(선택 여부 무관, 오검출 포함)
          - 굵은 원(초록=점등/회색=꺼짐) + R/Y/L/S 라벨 + 밝기값: 배치검사를 통과해
            실제 판정에 쓰인 4개(chosen이 None이면 이 단계까지 못 왔다는 뜻)
          - 좌상단 텍스트 3줄: 현재 인식 상태 / ROI 픽셀좌표 / 원검출 개수+실패사유
        DEBUG_VIZ_SIGNAL=True일 때 detect_s2()에서만 호출된다."""
        s = SIG4_VIZ_SCALE
        vis = cv2.resize(self.roi, None, fx=s, fy=s, interpolation=cv2.INTER_NEAREST)

        if all_circles is not None:
            for x, y, r in all_circles:
                cv2.circle(vis, (x * s, y * s), r * s, (0, 255, 255), 1, cv2.LINE_AA)
                cv2.circle(vis, (x * s, y * s), 1, (0, 255, 255), -1, cv2.LINE_AA)

        if chosen is not None:
            for (x, y, r), label, lit, bv in zip(chosen, SIG4_LABELS, self.s2_lit, self.s2_brightness):
                color = (0, 255, 0) if lit else (150, 150, 150)
                cv2.circle(vis, (x * s, y * s), r * s, color, 2, cv2.LINE_AA)
                cv2.putText(vis, f'{label}:{bv:.0f}', (x * s - r * s, y * s - r * s - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

        state = ('LEFT' if self.left_on else
                 'STR'  if self.straight_on else
                 'RED'  if self.red_on else '---')
        state_color = ((0, 255, 0) if state in ('LEFT', 'STR') else
                        (0, 0, 255) if state == 'RED' else (180, 180, 180))
        t, b, l, r_ = self.s2_roi_px
        reason = self.s2_reject_reason or 'OK'
        cv2.putText(vis, f'STATE:{state}', (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, state_color, 1, cv2.LINE_AA)
        cv2.putText(vis, f'roi=({t},{b},{l},{r_})', (4, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(vis, f'n={self.s2_circle_count} reason={reason}', (4, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

        self.vis = vis
        cv2.imshow('signal4_roi', self.vis)
        cv2.waitKey(1)

    def _draw_debug_board_search(self, frame, roi_boxes, chosen_idx):
        """'signal4_board_search' 창: 전체 프레임 위에 ①탐색범위(SIG4_SEARCH_ROI_*, 회색)
        ②이번 프레임에 시도한 후보 전부(자동탐색=주황, 마지막 고정폴백=파랑)
        ③채택된 후보(있으면 초록 굵은 테두리)를 겹쳐 그린다. _board_candidates()가 흰
        배경판을 실제로 얼마나 잘/잘못 찾는지 한눈에 보려는 순수 디버그 창(판단 로직에는
        영향 없음) — DEBUG_VIZ_SIGNAL=True일 때 detect_s2()에서만 호출된다."""
        h, w = frame.shape[:2]
        vis = frame.copy()

        st, sb = int(h * SIG4_SEARCH_ROI_T), int(h * SIG4_SEARCH_ROI_B)
        sl, sr = int(w * SIG4_SEARCH_ROI_L), int(w * SIG4_SEARCH_ROI_R)
        cv2.rectangle(vis, (sl, st), (sr, sb), (120, 120, 120), 1, cv2.LINE_AA)

        n_auto = len(roi_boxes) - 1  # 마지막 하나는 항상 고정폴백
        for idx, (t, b, l, r_) in enumerate(roi_boxes):
            is_fallback = (idx == n_auto)
            is_chosen   = (idx == chosen_idx)
            # BGR(OpenCV) 순서 — 초록=채택, 주황=고정폴백, 시안=자동후보
            color = (0, 255, 0) if is_chosen else ((0, 140, 255) if is_fallback else (255, 255, 0))
            thick = 3 if is_chosen else 1
            cv2.rectangle(vis, (l, t), (r_, b), color, thick, cv2.LINE_AA)

        status = f'chosen=#{chosen_idx}' if chosen_idx >= 0 else 'chosen=NONE(all failed)'
        cv2.putText(vis, f'board candidates: {n_auto} auto + 1 fallback | {status}',
                    (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(vis, 'gray=search band  cyan=auto candidate  orange=fixed fallback  green=chosen',
                    (4, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow('signal4_board_search', vis)
        cv2.waitKey(1)
