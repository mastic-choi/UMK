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
from .frst import frst_multiscale, find_peaks
from ..config import (
    SIG4_ROI_T, SIG4_ROI_B, SIG4_ROI_L, SIG4_ROI_R,
    SIG4_MIN_RADIUS, SIG4_MAX_RADIUS, SIG4_BRIGHT_MARGIN, SIG4_MAX_CANDIDATES,
    SIG4_GAP_UNIFORMITY_MAX_RATIO,
    SIG4_AUTOCROP_ENABLE, SIG4_SEARCH_ROI_T, SIG4_SEARCH_ROI_B,
    SIG4_SEARCH_ROI_L, SIG4_SEARCH_ROI_R,
    SIG4_BOARD_SAT_MAX, SIG4_BOARD_V_MIN, SIG4_BOARD_V_MAX,
    SIG4_BOARD_HUE_MIN, SIG4_BOARD_HUE_MAX, SIG4_BOARD_HUE_STD_MAX,
    SIG4_BOARD_MIN_AREA_PX, SIG4_BOARD_MAX_AREA_PX, SIG4_BOARD_MIN_ASPECT,
    SIG4_BOARD_MAX_ASPECT, SIG4_BOARD_MAX_CANDIDATES, SIG4_BOARD_CROP_MARGIN,
    SIG4_CIRCLE_ENGINE, SIG4_FRST_RADIUS_STEP, SIG4_FRST_ALPHA,
    SIG4_FRST_STD_FACTOR, SIG4_FRST_REL_THRESHOLD, SIG4_FRST_PEAK_MIN_DIST,
    SIG4_HOUGH_PARAM1, SIG4_HOUGH_PARAM2,
    SIG4_UNSHARP_ENABLE, SIG4_UNSHARP_AMOUNT, SIG4_UNSHARP_SIGMA,
    SIG4_CIRCLE_SAT_MAX,
    SIG4_FALLBACK_MAX_BOARD_FRAC,
    DEBUG_VIZ_SIGNAL, DEBUG_VIZ_SIGNAL_DETAIL,
    YOLO_SIGNAL_ENABLE, YOLO_SIGNAL_MAX_CANDIDATES, YOLO_SIGNAL_CROP_MARGIN,
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
    def __init__(self, yolo_detector=None):
        """yolo_detector: perception/yolo_signal.py의 YoloSignalDetector 인스턴스(선택).
        None이면(기본값, signal_offline_check.py 등 기존 호출부와 100% 호환) YOLO_SIGNAL_ENABLE
        여부와 무관하게 기존 HSV 자동크롭(_board_candidates())만 쓴다. track_drive.py가
        YOLO_SIGNAL_ENABLE=True일 때만 실제 인스턴스를 만들어 넘겨준다(yolo_signal_detection_plan.md
        참고) — 모델 파일이 아직 없어도(파일럿만 있는 상태) 이 클래스 자체는 항상 정상 동작한다."""
        self._yolo_detector = yolo_detector

        self.red_on = False
        self.straight_on = False
        self.left_on = False

        self.roi = None
        self.vis = None

        # ── [진단] track_drive._print_debug() 가 S0_SIGNAL 상태에서 읽는 값들 ──
        #   신호등 인식이 '어느 단계에서' 막혔는지 로그로 좁히기 위한 것.
        #   출발/교차로가 동일한 detect_s2()를 재사용하고(대회 규정 변경) 미션 state도
        #   S0_SIGNAL 하나로 통합됐으므로(2026-08-20) 진단 필드도 하나로 통합.
        #   이 속성들이 없으면 START_STATE 를 S0_SIGNAL 로 되돌리는 순간
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

    def circle_saturation(self, hsv, x, y, r):
        """[2026-08-18, §1.9] 원 내부 평균 채도(S). RAW 실측(신호등 rig 근접 캡처, lap_001보다
        블러 적음) 결과 진짜 램프 원은 S가 거의 항상 34 이하(회백색 렌즈/기판)인 반면, 천장
        조명 크롭에서 Hough가 잘못 잡은 원은 배경(사람 옷/잡다한 물체)이 섞여 S가 40~96까지
        자주 튄다 — board 블롭 단위 Hue검사(§1.4~§1.6)로 못 거른 걸 원 단위로 한 번 더 거르기
        위함."""
        y0, y1 = max(0, y - r // 2), y + r // 2
        x0, x1 = max(0, x - r // 2), x + r // 2
        patch = hsv[y0:y1, x0:x1, 1]

        if patch.size == 0:
            return 0.0
        return float(np.mean(patch))

    def shape_ok(self, circles, vert_max, horiz_max, min_dist, max_gap_ratio=SIG4_GAP_UNIFORMITY_MAX_RATIO):
        """배치 검사(원 정확히 4개 대상). 반환 (통과여부, 실패사유) — 사유는 디버그 로그용, 통과 시 ''.
        4개보다 많이 잡힌 경우의 완화 처리는 pick_best_4()가 담당. 4개 미만(가림/블러로 실제
        누락)은 이 함수까지 안 오고 detect_s2()에서 바로 실패 처리됨 — 없는 원을 만들어낼 근거가
        없기 때문. 프레임 단위 디바운스는 track_drive.py의 perc_signal()에서 처리.

        [2026-08-18] 등간격 검사 추가 — 실제 4구 신호등은 물리적으로 동일 간격이라, 인접 3개
        간격 중 최대/최소 비율이 너무 크면(다른 물체가 하나 섞였을 가능성) 개별 간격 조건을
        다 만족해도 실패 처리한다. config.py SIG4_GAP_UNIFORMITY_MAX_RATIO 주석의 실측
        근거 참고(정탐 1.13~1.64 vs 오염/오탐 2.22~2.64)."""
        xs = sorted(int(c[0]) for c in circles)
        ys = sorted(int(c[1]) for c in circles)

        if (ys[-1] - ys[0]) > vert_max:
            return False, f'vert_spread={ys[-1] - ys[0]}>{vert_max}'
        if (xs[-1] - xs[0]) > horiz_max:
            return False, f'horiz_spread={xs[-1] - xs[0]}>{horiz_max}'

        gaps = []
        for i in range(len(xs) - 1):
            gap = xs[i + 1] - xs[i]
            if gap < min_dist:
                return False, f'gap[{i}]={gap}<{min_dist}'
            gaps.append(gap)

        gap_ratio = max(gaps) / min(gaps)
        if gap_ratio > max_gap_ratio:
            return False, f'gap_ratio={gap_ratio:.2f}>{max_gap_ratio}(gaps={gaps})'
        return True, ''

    def pick_best_4(self, circles, vert_max, horiz_max, min_dist, max_gap_ratio=SIG4_GAP_UNIFORMITY_MAX_RATIO):
        """원이 4개보다 많이 잡혔을 때(반사광 등 오검출 섞임), 신호등 배치(shape_ok)를
        통과하는 4개 조합 중 가장 그럴듯한 것을 고른다.
        여러 조합이 통과하면 세로 퍼짐(vert_spread)이 가장 작은 걸 택한다 — 같은 높이에
        나란히 있을수록 진짜 신호등일 확률이 높다는 가정. 반환 (선택된 4개 또는 None, 실패사유)."""
        best, best_score = None, None
        for combo in itertools.combinations(circles, 4):
            ok, _ = self.shape_ok(combo, vert_max, horiz_max, min_dist, max_gap_ratio)
            if not ok:
                continue
            ys = [c[1] for c in combo]
            score = max(ys) - min(ys)
            if best_score is None or score < best_score:
                best, best_score = combo, score

        if best is None:
            return None, 'no_valid_4subset'
        return list(best), ''

    def _dominant_blob_frac(self, crop):
        """[2026-08-18, §1.10] crop 안에서 배경판 후보 마스크(HSV Hue/S/V, _board_candidates()와
        동일 기준)를 적용했을 때 가장 큰 연결 블롭 하나가 crop 면적의 몇 %를 차지하는지 반환.
        고정폴백(SIG4_ROI_*)은 지금까지 이 마스크/컨투어 검사를 안 거치고(auto 후보만 검사)
        무조건 시도됐는데, RAW(사용자 제공 근접 캡처) vs lap_001 실측 결과 이 값으로 "고정폴백
        위치에 지금 진짜 배경판이 있는지"를 가른다 — 진짜 배경판은 크롭 안에서 여백을 두고
        적당히만 차지하는데(7.5~12.2%), 천장 형광등 패널처럼 넓게 이어진 저채도 광원이 크롭
        대부분을 채우면 13.2~38.5%까지 올라간다. 원 단위 특징(위치/채도/엣지, §1.6~§1.9)은
        전부 두 데이터셋 사이에서 안 옮겨갔지만, 이 "크롭 대비 블롭 점유율"은 두 데이터셋
        모두에서 진짜 범위와 겹치지 않았다."""
        if crop.size == 0:
            return 1.0
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (SIG4_BOARD_HUE_MIN, 0, SIG4_BOARD_V_MIN),
                            (SIG4_BOARD_HUE_MAX, SIG4_BOARD_SAT_MAX, SIG4_BOARD_V_MAX))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return 0.0
        largest = max(cv2.contourArea(c) for c in contours)
        return largest / (crop.shape[0] * crop.shape[1])

    def _board_candidates(self, frame):
        """흰 배경판(신호등 rig) 후보 사각형을 탐색범위(SIG4_SEARCH_ROI_*) 안에서 찾아
        (t,b,l,r) 절대 픽셀좌표 리스트로 반환한다(마진 포함, 프레임 경계로 클리핑됨,
        면적 작은 순 정렬 — 보드에 더 타이트하게 맞는 후보부터 시도). 이 함수는 "후보
        제안"만 하고, 그중 진짜 신호등인지는 detect_s2()의 기존 원 4개 패턴검사
        (find_circles/shape_ok/pick_best_4)가 최종 판정한다. 실패해도 예외 없이 빈
        리스트만 반환 — 호출부가 SIG4_ROI_* 고정 폴백으로 이어가면 된다.

        [2026-08-18] 그레이스케일 밝기 단일 임계값은 lap_001 실차 캡처에서 거의 항상
        실패했다(배경판과 블러로 번진 천장 형광등이 하나의 거대한 블롭으로 뭉개짐,
        config.py 주석 참고) — HSV로 바꿔 채도(S)를 낮게 제한하니(무채색=배경판 특징)
        분리가 됐다. 채도만으론 완전하지 않아(형광등 주변도 저채도인 경우가 있음)
        면적 상한을 훨씬 타이트하게 잡고, MORPH_OPEN으로 형광등 반사가 만드는 가늘고
        긴 줄무늬를 끊어낸다(MORPH_CLOSE는 오히려 그런 줄무늬를 서로 이어붙여 역효과).

        [2026-08-18 2차] Hue(H) 밴드도 추가. lap_001 실측 히스토그램(config.py
        SIG4_BOARD_HUE_MIN/MAX 주석 참고)상 진짜 배경판은 H가 좁게(80~110대) 몰리는데
        예전 오탐 블롭(frame_062)은 훨씬 넓게 퍼져 있어, 이 범위 밖 픽셀을 마스크
        단계에서 미리 제외하면 이종 색상이 섞인 병합 블롭을 더 작게/더 쪼개서 잡을
        가능성이 있다 — 정상 프레임 손실은 실측상 1% 미만.

        [2026-08-18 3차] 블롭 단위 Hue 표준편차(균질성) 필터 추가. §1.4/§1.5(README) 실험 중
        면적/위치로는 오염된 블롭(진짜 배경판+엉뚱한 물체가 병합)과 깨끗한 블롭을 못 갈랐지만,
        Hue 표준편차는 갈렸다 — lap_001 chosen 후보 기준 정상(52/53/55/56) 16.5~19.8인데
        예전 오탐(frame_062, 큰 블롭이 쪼개진 뒤에도) 25.5~30.1로 뚜렷이 높았다(config.py
        SIG4_BOARD_HUE_STD_MAX 주석 참고). 색상이 여러 물체가 섞여 들쭉날쭉한 블롭을 여기서
        미리 걸러 원 4개 패턴검사(circle_count/shape_ok)까지 가지도 못하게 한다 — 등간격검사
        (§1.3)가 "원의 배치"로 걸러내는 오탐을, 이건 "블롭의 색상 균질성"으로 한 단계 앞에서
        거른다는 점에서 서로 다른 근거의 독립적 필터."""
        h, w = frame.shape[:2]
        st, sb = int(h * SIG4_SEARCH_ROI_T), int(h * SIG4_SEARCH_ROI_B)
        sl, sr = int(w * SIG4_SEARCH_ROI_L), int(w * SIG4_SEARCH_ROI_R)
        band = frame[st:sb, sl:sr]
        if band.size == 0:
            return []

        hsv = cv2.cvtColor(band, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (SIG4_BOARD_HUE_MIN, 0, SIG4_BOARD_V_MIN),
                            (SIG4_BOARD_HUE_MAX, SIG4_BOARD_SAT_MAX, SIG4_BOARD_V_MAX))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

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

            # 블롭(컨투어) 내부 픽셀만의 Hue 표준편차 — bbox 전체가 아니라 mask(색상 조건)를
            # 통과한 픽셀만 대상으로 해야 배경(마진)에 섞여드는 걸 피할 수 있다.
            local_mask = np.zeros((ch, cw), dtype=np.uint8)
            cv2.drawContours(local_mask, [c], -1, 255, -1, offset=(-x, -y))
            local_mask = cv2.bitwise_and(local_mask, mask[y:y + ch, x:x + cw])
            hue_vals = hsv[y:y + ch, x:x + cw, 0][local_mask > 0]
            if hue_vals.size == 0 or float(hue_vals.std()) > SIG4_BOARD_HUE_STD_MAX:
                continue

            mx, my = int(cw * SIG4_BOARD_CROP_MARGIN), int(ch * SIG4_BOARD_CROP_MARGIN)
            # band-로컬 좌표 → 원본 frame 절대좌표 변환 + 마진 + 프레임 경계 클리핑
            l  = max(0, sl + x - mx)
            r_ = min(w, sl + x + cw + mx)
            t  = max(0, st + y - my)
            b  = min(h, st + y + ch + my)
            scored.append((area, t, b, l, r_))

        scored.sort(key=lambda v: v[0])  # 작은 것부터 — 보드 크기에 더 타이트하게 맞는 후보 우선
        return [(t, b, l, r_) for _, t, b, l, r_ in scored[:SIG4_BOARD_MAX_CANDIDATES]]

    def _board_candidates_yolo(self, frame):
        """YOLO(self._yolo_detector)로 배경판 후보를 찾아 (t,b,l,r) 절대 픽셀좌표 리스트로
        반환 — _board_candidates()(HSV 자동크롭)의 대체 소스. 신뢰도 내림차순 정렬 후 상위
        YOLO_SIGNAL_MAX_CANDIDATES개만 쓰고, 각 bbox에 YOLO_SIGNAL_CROP_MARGIN만큼 여유를
        더해 프레임 경계로 클리핑한다(_board_candidates()의 SIG4_BOARD_CROP_MARGIN과 동일한
        이유 — 원이 bbox 경계에 안 걸리게). 그 뒤 원 4개 패턴검사(find_circles*/shape_ok/
        pick_best_4)는 무수정으로 그대로 재사용한다 — detect_s2()가 이 함수와
        _board_candidates()를 완전히 동일한 반환 형식으로 취급하기 때문."""
        h, w = frame.shape[:2]
        detections = self._yolo_detector.detect(frame)
        detections.sort(key=lambda d: d[4], reverse=True)

        boxes = []
        for x1, y1, x2, y2, _conf in detections[:YOLO_SIGNAL_MAX_CANDIDATES]:
            bw, bh = x2 - x1, y2 - y1
            mx, my = bw * YOLO_SIGNAL_CROP_MARGIN, bh * YOLO_SIGNAL_CROP_MARGIN
            t = max(0, int(y1 - my))
            b = min(h, int(y2 + my))
            l = max(0, int(x1 - mx))
            r_ = min(w, int(x2 + mx))
            boxes.append((t, b, l, r_))
        return boxes

    def find_circles(self, roi, min_r, max_r):
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        if SIG4_UNSHARP_ENABLE:
            soft = cv2.GaussianBlur(blur, (0, 0), SIG4_UNSHARP_SIGMA)
            blur = cv2.addWeighted(blur, 1 + SIG4_UNSHARP_AMOUNT, soft, -SIG4_UNSHARP_AMOUNT, 0)

        circles = cv2.HoughCircles(
            blur, cv2.HOUGH_GRADIENT, 1, 20,
            param1=SIG4_HOUGH_PARAM1, param2=SIG4_HOUGH_PARAM2,
            minRadius=min_r, maxRadius=max_r
        )

        return gray, circles

    def _frst_band_peaks(self, frame, min_r, max_r):
        """FRST(perception/frst.py)를 SIG4_SEARCH_ROI_* 전체 범위에 대해 딱 한 번만 계산해,
        피크 전부를 프레임 절대좌표 (x,y,r) 리스트로 반환한다.

        [중요, 2026-08-18 버그 수정] 처음엔 후보 크롭마다 매번 FRST를 새로 돌렸는데,
        find_peaks()의 임계값(SIG4_FRST_REL_THRESHOLD)이 "그 크롭 자체의 최댓값" 대비
        상대값이라 — 크롭을 작게 자를수록 그 안에 진짜 램프가 없어도 "크롭 안에서
        상대적으로 제일 강한 지점"이 항상 임계값을 넘어버렸다. lap_001 실측(70프레임 중
        44개가 "성공"으로 나옴, 대부분 천장조명 오탐)으로 발견됨. 고쳐서, 임계값을
        "탐색범위 전체의 최댓값" 대비로 한 번만 계산하고, 크롭별로는 그 절대적인 피크
        목록 중 자기 범위 안에 있는 것만 걸러 쓴다 — 작은 크롭이라고 기준이 느슨해지지
        않는다. find_circles_frst()가 이 결과를 각 후보 박스 범위로 필터링해 쓴다."""
        h, w = frame.shape[:2]
        st, sb = int(h * SIG4_SEARCH_ROI_T), int(h * SIG4_SEARCH_ROI_B)
        sl, sr = int(w * SIG4_SEARCH_ROI_L), int(w * SIG4_SEARCH_ROI_R)
        band = frame[st:sb, sl:sr]
        gray = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY)

        radii = range(min_r, max_r + 1, SIG4_FRST_RADIUS_STEP)
        sym_map, radius_map = frst_multiscale(
            gray, radii, alpha=SIG4_FRST_ALPHA, std_factor=SIG4_FRST_STD_FACTOR)
        peaks = find_peaks(
            sym_map, radius_map,
            max_peaks=SIG4_MAX_CANDIDATES * 3,  # 여러 박스에 나눠 걸릴 수 있으니 넉넉히
            min_dist=SIG4_FRST_PEAK_MIN_DIST,
            rel_threshold=SIG4_FRST_REL_THRESHOLD)

        return [(x + sl, y + st, r) for x, y, r in peaks]  # band-로컬 → 프레임 절대좌표

    def find_circles_frst(self, roi, roi_box, band_peaks_abs):
        """find_circles()의 FRST 버전 — band_peaks_abs(_frst_band_peaks()가 전체 범위에서
        한 번만 계산한 절대좌표 피크 목록)를 roi_box 범위로 필터링하고, roi_box 기준
        상대좌표로 변환해 cv2.HoughCircles()와 동일한 (gray, circles) 형식으로 반환한다
        — detect_s2() 이후 로직(shape_ok/pick_best_4)을 전혀 안 건드리고 재사용하기 위함.
        lap_001 실측으로 Hough보다 후보 오염(천장조명이 진짜 램프와 섞이는 문제)이 덜한
        것으로 확인돼 SIG4_CIRCLE_ENGINE 기본값이 됐다(config.py 해당 주석 참고)."""
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        t, b, l, r_ = roi_box
        local = [(x - l, y - t, r) for x, y, r in band_peaks_abs
                 if l <= x < r_ and t <= y < b]
        if not local:
            return gray, None
        circles = np.array([local], dtype=np.float64)  # HoughCircles와 동일하게 (1,N,3) 형태
        return gray, circles

    def detect_s2(self, frame):
        """4구 신호등 인식 — S0(출발)/S2(교차로) 공통.
          S0에서 쓸 때: straight_on(초록만 점등) = 출발
          S2에서 쓸 때: straight_on = 직진, left_on(초록+빨강 동시) = 좌회전

        [2026-08-18] ROI를 SIG4_ROI_*(고정 크롭) 하나가 아니라 "후보 리스트"로 순회한다 —
        SIG4_AUTOCROP_ENABLE=True면 _board_candidates()가 흰 배경판(신호등 rig) 후보를 먼저
        제안하고, 그 뒤에 항상 기존 SIG4_ROI_*(고정 크롭)를 마지막 안전망으로 덧붙인다. 각
        후보에 대해 아래 원 4개 패턴검사(find_circles*→shape_ok/pick_best_4, 판정 로직 자체는
        그대로)를 순서대로 시도해 맨 처음 성공하는 후보를 채택하고 즉시 멈춘다 — 자동탐색이
        전부 실패해도 도입 전과 똑같이 고정 크롭으로 마지막에 한 번 더 시도되므로 이 변경으로
        인식률이 이전보다 나빠지진 않는다. 크롭 안에서 원을 찾는 엔진은
        SIG4_CIRCLE_ENGINE(config.py)로 Hough Circle('hough')과 FRST('frst', 기본값) 중
        고를 수 있다 — find_circles()/find_circles_frst() 둘 다 같은 (gray, circles) 형식을
        반환하므로 이 아래 로직은 어느 쪽을 쓰든 동일하다.

        원이 정확히 4개가 아니어도 어느 정도는 견딘다(반사광 등으로 여분의 원이 섞이는 경우가
        실차에서 흔해서): 4개 초과면 pick_best_4()로 가장 신호등다운 4개 조합을 고른다.
        4개 미만이면(가림/블러로 실제로 못 찾은 경우) 만들어낼 근거가 없으므로 그 후보는 실패
        처리하고 다음 후보로 넘어간다.
        프레임 단위 디바운스(연속 N프레임 확정)는 track_drive.py의 perc_signal()에서 처리한다
        (여기는 항상 '이번 한 프레임' 기준 순간값만 반환)."""
        if frame is None:
            return self.red_on, self.straight_on, self.left_on

        h, w = frame.shape[:2]

        # [2026-08-19] 배경판 후보 소스 선택 — YOLO_SIGNAL_ENABLE=True고 yolo_detector가
        # 주어졌으면 YOLO(_board_candidates_yolo())로 대체, 아니면 기존 HSV 자동크롭
        # (_board_candidates())을 그대로 쓴다. 어느 쪽이든 그 뒤 원 4개 패턴검사 로직은
        # 완전히 동일 — 아래 fixed 고정폴백도 소스와 무관하게 항상 마지막에 붙는다.
        if YOLO_SIGNAL_ENABLE and self._yolo_detector is not None:
            roi_boxes = list(self._board_candidates_yolo(frame))
        elif SIG4_AUTOCROP_ENABLE:
            roi_boxes = list(self._board_candidates(frame))
        else:
            roi_boxes = []
        fixed_t, fixed_b = int(h * SIG4_ROI_T), int(h * SIG4_ROI_B)
        fixed_l, fixed_r = int(w * SIG4_ROI_L), int(w * SIG4_ROI_R)
        # [2026-08-18, §1.10] 고정폴백도 _dominant_blob_frac()로 "지금 이 위치에 진짜 배경판
        # 다운 블롭이 있는지" 확인한 뒤에만 후보로 추가한다 — 넘으면(천장조명 등이 크롭 대부분을
        # 채운 경우) 안전망 자체를 포기하고 이 프레임은 정직하게 실패 처리한다.
        fixed_frac = self._dominant_blob_frac(frame[fixed_t:fixed_b, fixed_l:fixed_r])
        if fixed_frac <= SIG4_FALLBACK_MAX_BOARD_FRAC:
            roi_boxes.append((fixed_t, fixed_b, fixed_l, fixed_r))

        self.s2_board_candidates = roi_boxes
        self.s2_chosen_idx = -1

        if not roi_boxes:
            self.s2_roi_px        = (fixed_t, fixed_b, fixed_l, fixed_r)
            self.s2_circle_count  = 0
            self.s2_reject_reason = f'no_board_candidates(fixed_frac={fixed_frac:.2f}>{SIG4_FALLBACK_MAX_BOARD_FRAC})'
            self.s2_brightness    = []
            self.s2_lit           = []
            self.red_on = self.straight_on = self.left_on = False
            return self.red_on, self.straight_on, self.left_on

        self.red_on = self.straight_on = self.left_on = False
        all_circles = chosen = gray = None

        use_frst = SIG4_CIRCLE_ENGINE == 'frst'
        # FRST는 후보 박스마다 새로 돌리지 않고 SIG4_SEARCH_ROI_* 전체에 대해 프레임당 한 번만
        # 계산한다 — find_circles_frst() 주석 참고("크롭마다 새로 돌리면 임계값이 크롭 자체
        # 기준 상대값이 돼버려 작은 크롭일수록 오탐이 늘어나는" 버그를 막기 위함.
        band_peaks = self._frst_band_peaks(frame, SIG4_MIN_RADIUS, SIG4_MAX_RADIUS) if use_frst else None

        for idx, (t, b, l, r_) in enumerate(roi_boxes):
            roi = frame[t:b, l:r_]
            if use_frst:
                gray, raw_circles = self.find_circles_frst(roi, (t, b, l, r_), band_peaks)
            else:
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

                if SIG4_CIRCLE_SAT_MAX is not None:
                    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                    kept = [c for c in all_circles
                            if self.circle_saturation(hsv_roi, c[0], c[1], c[2]) <= SIG4_CIRCLE_SAT_MAX]
                    all_circles = np.array(kept, dtype=int) if kept else np.empty((0, 3), dtype=int)
                    self.s2_circle_count = len(all_circles)

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

        if DEBUG_VIZ_SIGNAL_DETAIL:
            self._draw_debug_viz(all_circles, chosen)
            self._draw_debug_board_search(frame, roi_boxes, self.s2_chosen_idx)

        return self.red_on, self.straight_on, self.left_on

    def _draw_debug_viz(self, all_circles, chosen):
        """'signal4_roi' 창: ROI 크롭을 SIG4_VIZ_SCALE배 확대해 그 위에
          - 노란 원: Hough가 찾은 원 전부(선택 여부 무관, 오검출 포함)
          - 굵은 원(초록=점등/회색=꺼짐) + R/Y/L/S 라벨 + 밝기값: 배치검사를 통과해
            실제 판정에 쓰인 4개(chosen이 None이면 이 단계까지 못 왔다는 뜻)
          - 좌상단 텍스트 3줄: 현재 인식 상태 / ROI 픽셀좌표 / 원검출 개수+실패사유
        DEBUG_VIZ_SIGNAL_DETAIL=True일 때 detect_s2()에서만 호출된다."""
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
        영향 없음) — DEBUG_VIZ_SIGNAL_DETAIL=True일 때 detect_s2()에서만 호출된다."""
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
