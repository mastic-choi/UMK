import math
import cv2
import numpy as np

#=============================================
# xycar_application/app_hough_drive/app_hough_drive/app_hough_drive.py 의
# 차선 추출 로직(ROI 크롭 → Canny 엣지 → HoughLinesP → 기울기/위치로 좌우 분류
# → 평균 직선 산출 → 기준행과의 교점으로 차선 위치 추정)을 track_drive의
# LaneDetector 인터페이스에 맞게 이식한 모듈. BEV/색상마스크/moments는 쓰지 않는다.
#
# app_hough_drive.py는 기준행(L_ROW) 하나로만 조향각을 바로 계산했지만,
# track_drive의 PID/코너 예측 감속 로직은 근거리(offset)·원거리(lookahead)
# 편차 두 개가 필요하므로 기준행을 근거리/원거리 두 개로 늘려 계산한다.
# 그 외 로직(기울기부호+화면위치로 좌/우 분류, 대표직선 산출, 한쪽 차선만
# 검출됐을 때 반대쪽을 도로폭으로 추정, 둘 다 놓쳤을 때 직전값 유지)은
# 원본과 동일하다.
#=============================================

# ROI(세로 구간) — 중심(≈300px)은 app_hough_drive.py 원본(300~380, 80px 밴드)과 거의
# 같게, 밴드 높이는 73px로 좁혀서 쓴다.
#   ※ track_drive는 app_hough_drive.py(xycar_cam)와 다른 카메라(usb_cam 전방)를
#     쓰므로 이 ROI가 실제로 노면만 담는지 실차에서 반드시 재확인·재튜닝할 것.
HOUGH_ROI_TOP = 264 / 480
HOUGH_ROI_BOT = 336 / 480

# Canny / HoughLinesP 파라미터.
GAUSSIAN_KSIZE = 5
CANNY_LOW = 60
CANNY_HIGH = 75
HOUGH_RHO = 1
HOUGH_THETA = math.pi / 180
HOUGH_THRESHOLD = 40
HOUGH_MIN_LINE_LEN = 30
HOUGH_MAX_LINE_GAP = 20

# 기울기 절대값이 이 값 이하면 수평선(차선 아님)으로 보고 제외 — 원본과 동일.
SLOPE_MIN = 0.2

# 짧은 직선(노면 이음새·반사·잡음 등으로 생기는 오검출) 제거.
#   실제 차선은 길게 이어지는 반면 잡음은 대체로 짧게 끊어져 잡히므로, HoughLinesP가
#   반환한 선분 중 이 길이(px) 미만은 좌/우 분류·대표직선 계산에서 아예 제외한다.
MIN_LANE_LINE_LEN = 45

# 선분 병합(그룹화) — HoughLinesP는 MAX_LINE_GAP(끊긴 구간 허용치) 때문에 실제로는
#   하나로 이어진 차선도 여러 조각으로 쪼개서 반환하는 경우가 많다. 조각 하나하나의
#   개별 길이만으로 평가하면 조각난 진짜 차선이 안 끊긴 짧은 잡음보다 오히려 손해를
#   볼 수 있다. 그래서 기준행(ROI 중간)에서의 x위치가 서로 이 값(px) 이내로 가까운
#   선분들을 하나의 그룹으로 묶고, "그룹의 총 길이"가 가장 큰 그룹만 그 쪽의 대표
#   차선 후보로 채택한다.
GROUP_X_TOL = 25.0

# 기준행 위치 (ROI 세로비율, 0=원거리/ROI 위쪽 ~ 1=근거리/ROI 아래쪽).
#   원본은 L_ROW=40/ROI_HEIGHT=80 로 정확히 중간(0.5) 한 지점만 썼다.
#   여기서는 near/far 두 지점으로 나누되, ROI 경계에 너무 붙으면(외삽 오차 커짐)
#   Hough 선분이 짧아 대표직선이 불안정해지므로 가장자리에서 조금 띄웠다.
NEAR_ROW_RATIO = 0.80
FAR_ROW_RATIO = 0.20

# 명시적 경로(웨이포인트) — lane_util.SlideWindow와 인터페이스를 맞추기 위해 이
# 백엔드도 path를 반환한다. 다만 Hough는 애초에 대표직선(m,b) 하나만 피팅하므로
# 여기서 만드는 path도 근거리/원거리 두 점을 잇는 "직선"을 균등 샘플링한 것일 뿐,
# lane_util처럼 슬라이스별 점을 선형보간으로 잇는 건 아니다(이 백엔드의 구조적
# 한계 — 커브 추종 품질이 중요하면 dl/classic_cv 백엔드를 쓸 것).
HOUGH_PATH_N_WAYPOINTS = 6

# 한쪽 차선만 검출됐을 때 반대쪽 위치 추정에 쓰는 도로폭(px) 초기값 및 그 후 EMA 갱신.
#   원본 app_hough_drive.py의 "x_left = x_right - 380"/"x_right = x_left + 380" 그대로.
#   실차 재검증 필요(카메라 마운트가 다르면 이 폭도 다시 재봐야 함).
DEFAULT_LANE_WIDTH = 380.0
LANE_WIDTH_MIN, LANE_WIDTH_MAX = 180.0, 400.0
LANE_WIDTH_EMA_ALPHA = 0.1

# 노란 중앙선 위치 추정용 색상 마스크 — Hough 로직 자체는 색상을 쓰지 않지만,
# obstacle_avoidance의 회피방향 판단(choose_side)이 lane_side(내가 어느 차선에
# 있는지)에 의존하므로, 그 판정에 필요한 최소한의 정보만 별도로 뽑아 제공한다.
# dl_lane.py와 값을 공유하므로 config.py에서 가져온다.
# DEBUG_VIZ_HOUGH_LANE도 config.py에 있다 — 실차 테스트 중 값을 바꾸려면
# 이 파일이 아니라 config.py를 고칠 것.
from ..config import (
    YELLOW_LOWER, YELLOW_UPPER, DEBUG_VIZ_HOUGH_LANE,
    YELLOW_DASH_ROI_TOP, YELLOW_DASH_ROI_BOT, YELLOW_DASH_ROI_X0, YELLOW_DASH_ROI_X1,
    YELLOW_DASH_MIN_AREA_PX, YELLOW_DASH_MAX_AREA_PX,
    YELLOW_DASH_PRESENT_FRAMES, YELLOW_DASH_ABSENT_FRAMES, DEBUG_VIZ_DASH_COUNTER,
    CHECKER_ROI_TOP, CHECKER_ROI_BOT, CHECKER_BLACK_MAX_V,
    CHECKER_DARK_RATIO_TH, CHECKER_YELLOW_RATIO_TH, CHECKER_CONFIRM_FRAMES,
    DEBUG_VIZ_CHECKER_GATE,
)

RED = (0, 0, 255)
YELLOW_COL = (0, 255, 255)
BLUE = (255, 0, 0)
GREEN = (0, 255, 0)


class HoughLaneDetector:
    """
    기존 LaneDetector(BEV+moments, perc_floor.py)와 동일한
        detect(frame) -> (lane_valid, lane_offset, lane_lookahead, lane_center, path, debug_img)
    인터페이스를 유지한다. track_drive.py의 perc_lane()은 수정 없이 그대로 재사용 가능.
    """

    def __init__(self):
        self.prev_x_left = None
        self.prev_x_right = None
        self.lane_width = DEFAULT_LANE_WIDTH
        self.roi_w = 0
        self.roi_h = 0
        # _update_lane_side()용 — [(y, x)] 또는 빈 리스트 (lane_util.SlideWindow.yellow_centers 형식과 호환)
        self.yellow_centers = []

    def detect(self, frame):
        if frame is None:
            lane_center = self.roi_w / 2.0 if self.roi_w else 0.0
            return False, 0.0, 0.0, lane_center, [], None

        h, w = frame.shape[:2]
        y0, y1 = int(h * HOUGH_ROI_TOP), int(h * HOUGH_ROI_BOT)
        roi_img = frame[y0:y1, 0:w]
        self.roi_h, self.roi_w = roi_img.shape[:2]
        debug_img = roi_img.copy()

        self._update_yellow_centers(roi_img)

        gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
        blur_gray = cv2.GaussianBlur(gray, (GAUSSIAN_KSIZE, GAUSSIAN_KSIZE), 0)
        edge_img = cv2.Canny(np.uint8(blur_gray), CANNY_LOW, CANNY_HIGH)

        all_lines = cv2.HoughLinesP(edge_img, HOUGH_RHO, HOUGH_THETA, HOUGH_THRESHOLD,
                                     HOUGH_MIN_LINE_LEN, HOUGH_MAX_LINE_GAP)
        if all_lines is None:
            return self._finish(False, 0.0, 0.0, debug_img, edge_img)

        # 기울기 부호(음수=좌, 양수=우) + 화면 위치로 좌/우 선분 분류.
        #   위치 판정은 반드시 중점(mid_x)으로 한다 — HoughLinesP는 한 선분의 두 끝점을
        #   (x1,y1)/(x2,y2) 어느 순서로 반환할지 보장하지 않는데, 원본 app_hough_drive.py처럼
        #   특정 한쪽 끝점(x2 또는 x1)만 검사하면 순서가 뒤집힌 프레임에서 유효한 차선이
        #   조건을 못 넘겨 누락된다. 이게 한쪽 차선의 검출 신뢰도를 떨어뜨려 조향이 그
        #   방향으로 쏠리는 원인이 될 수 있어(적분항과 결합 시 특히), 끝점 순서와 무관한
        #   중점 기준으로 바꿨다.
        left_lines, right_lines = [], []
        for line in all_lines:
            x1, ly1, x2, ly2 = line[0]
            if math.hypot(x2 - x1, ly2 - ly1) < MIN_LANE_LINE_LEN:
                continue   # 짧은 잡음 선분 제외
            slope = 1000.0 if x2 == x1 else float(ly2 - ly1) / float(x2 - x1)
            if abs(slope) <= SLOPE_MIN:
                continue
            mid_x = (x1 + x2) / 2.0
            if slope < 0 and mid_x < self.roi_w / 2.0:
                left_lines.append((x1, ly1, x2, ly2))
                cv2.line(debug_img, (x1, ly1), (x2, ly2), RED, 2)
            elif slope > 0 and mid_x > self.roi_w / 2.0:
                right_lines.append((x1, ly1, x2, ly2))
                cv2.line(debug_img, (x1, ly1), (x2, ly2), YELLOW_COL, 2)

        if not left_lines and not right_lines:
            return self._finish(False, 0.0, 0.0, debug_img, edge_img)

        # 조각난 선분들을 병합(그룹화)해서, 총 길이가 가장 긴 그룹(=진짜 차선일
        # 가능성이 가장 높은 쪽)만 대표직선 계산에 사용한다.
        left_lines = self._select_main_group(left_lines)
        right_lines = self._select_main_group(right_lines)
        for x1, ly1, x2, ly2 in left_lines:
            cv2.line(debug_img, (x1, ly1), (x2, ly2), GREEN, 3)
        for x1, ly1, x2, ly2 in right_lines:
            cv2.line(debug_img, (x1, ly1), (x2, ly2), GREEN, 3)

        left_fit = self._fit_line(left_lines)
        right_fit = self._fit_line(right_lines)

        near_row = self.roi_h * NEAR_ROW_RATIO
        far_row = self.roi_h * FAR_ROW_RATIO

        x_left_near, x_left_far = self._x_at(left_fit, near_row), self._x_at(left_fit, far_row)
        x_right_near, x_right_far = self._x_at(right_fit, near_row), self._x_at(right_fit, far_row)

        if x_left_near is None and x_right_near is not None:
            # 왼쪽 차선을 놓쳤으면 오른쪽 + 도로폭으로 추정 (원본과 동일한 방식)
            x_left_near = x_right_near - self.lane_width
            x_left_far = (x_right_far - self.lane_width) if x_right_far is not None else x_left_near
        elif x_right_near is None and x_left_near is not None:
            x_right_near = x_left_near + self.lane_width
            x_right_far = (x_left_far + self.lane_width) if x_left_far is not None else x_right_near
        elif x_left_near is None and x_right_near is None:
            # 둘 다 놓쳤으면 직전 프레임 값 유지 (원본의 prev_x_left/prev_x_right 폴백과 동일한 의도)
            if self.prev_x_left is None or self.prev_x_right is None:
                return self._finish(False, 0.0, 0.0, debug_img, edge_img)
            x_left_near, x_right_near = self.prev_x_left, self.prev_x_right
            x_left_far, x_right_far = x_left_near, x_right_near
        else:
            width = x_right_near - x_left_near
            if LANE_WIDTH_MIN < width < LANE_WIDTH_MAX:
                self.lane_width = (1 - LANE_WIDTH_EMA_ALPHA) * self.lane_width + LANE_WIDTH_EMA_ALPHA * width

        self.prev_x_left, self.prev_x_right = x_left_near, x_right_near

        near_center = (x_left_near + x_right_near) / 2.0
        far_center = (x_left_far + x_right_far) / 2.0

        offset = near_center - self.roi_w / 2.0
        lookahead = far_center - self.roi_w / 2.0
        lane_center = self.roi_w / 2.0 + offset

        # 근거리(near_row,near_center)-원거리(far_row,far_center) 두 점을 잇는 직선을
        # N개 점으로 선형보간해 웨이포인트로 만든다. 순서는 lane_util과 동일하게
        # 가까운점(near_row)→먼점(far_row)이 되도록 rows를 near_row에서 far_row로 스캔한다.
        rows = np.linspace(near_row, far_row, HOUGH_PATH_N_WAYPOINTS)
        t = (rows - near_row) / (far_row - near_row) if far_row != near_row else np.zeros_like(rows)
        xs = near_center + t * (far_center - near_center)
        path = list(zip(xs.tolist(), rows.tolist()))

        if DEBUG_VIZ_HOUGH_LANE:
            self._draw_debug(debug_img, near_row, x_left_near, x_right_near, near_center)

        return self._finish(True, offset, lookahead, debug_img, edge_img, lane_center, path)

    def _select_main_group(self, lines):
        """기준행(ROI 중간)에서의 x위치가 가까운 선분들을 그룹으로 묶고, 총 길이가
        가장 긴 그룹만 반환한다. HoughLinesP가 하나의 실제 차선을 여러 조각으로
        쪼개 반환해도 조각들을 다시 합쳐서 하나의 긴 차선으로 취급하기 위함이다.
        선분이 1개뿐이면 그대로 반환."""
        if len(lines) <= 1:
            return lines

        ref_row = self.roi_h / 2.0
        items = []
        for x1, y1, x2, y2 in lines:
            if x2 == x1:
                continue
            m = float(y2 - y1) / float(x2 - x1)
            if m == 0.0:
                continue
            b = y1 - m * x1
            x_ref = (ref_row - b) / m
            length = math.hypot(x2 - x1, y2 - y1)
            items.append((x_ref, length, (x1, y1, x2, y2)))

        if not items:
            return lines

        items.sort(key=lambda it: it[0])

        groups = [[items[0]]]
        for it in items[1:]:
            if it[0] - groups[-1][-1][0] <= GROUP_X_TOL:
                groups[-1].append(it)
            else:
                groups.append([it])

        best_group = max(groups, key=lambda g: sum(it[1] for it in g))
        return [it[2] for it in best_group]

    def _fit_line(self, lines):
        """선분들의 기울기·양끝점을 길이로 가중평균해 대표직선 (m, b)를 구한다.
        MIN_LANE_LINE_LEN을 넘겨 일단 채택된 선분이라도, 그중 더 긴(=진짜 차선일
        가능성이 높은) 선분이 결과에 더 크게 반영되도록 길이를 가중치로 쓴다.
        m==0(못 찾음)이면 None."""
        if not lines:
            return None
        x_sum = y_sum = m_sum = w_sum = 0.0
        for x1, y1, x2, y2 in lines:
            length = math.hypot(x2 - x1, y2 - y1)
            slope = (float(y2 - y1) / float(x2 - x1)) if x2 != x1 else 0.0
            w_sum += length
            x_sum += (x1 + x2) * length
            y_sum += (y1 + y2) * length
            m_sum += slope * length
        if w_sum == 0.0:
            return None
        m = m_sum / w_sum
        if m == 0.0:
            return None
        x_avg, y_avg = x_sum / (w_sum * 2), y_sum / (w_sum * 2)
        b = y_avg - m * x_avg
        return m, b

    def _x_at(self, fit, row):
        """대표직선 (m,b)와 y=row 수평선의 교점 x좌표. fit이 없으면 None."""
        if fit is None:
            return None
        m, b = fit
        return (row - b) / m

    def _update_yellow_centers(self, roi_img):
        hsv = cv2.cvtColor(roi_img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, YELLOW_LOWER, YELLOW_UPPER)
        ys, xs = np.nonzero(mask)
        if len(xs) == 0:
            self.yellow_centers = []
            return
        self.yellow_centers = [(float(np.mean(ys)), float(np.mean(xs)))]

    def _draw_debug(self, img, near_row, x_left, x_right, x_mid):
        y = int(near_row)
        cv2.line(img, (0, y), (self.roi_w, y), YELLOW_COL, 1)
        cv2.rectangle(img, (int(x_left) - 5, y - 5), (int(x_left) + 5, y + 5), GREEN, 2)
        cv2.rectangle(img, (int(x_right) - 5, y - 5), (int(x_right) + 5, y + 5), GREEN, 2)
        cv2.rectangle(img, (int(x_mid) - 5, y - 5), (int(x_mid) + 5, y + 5), BLUE, 2)
        cv2.rectangle(img, (self.roi_w // 2 - 5, y - 5), (self.roi_w // 2 + 5, y + 5), RED, 2)

    def _finish(self, valid, offset, lookahead, debug_img, edge_img, lane_center=None, path=None):
        if lane_center is None:
            lane_center = self.roi_w / 2.0 if self.roi_w else 0.0
        if path is None:
            path = []
        if DEBUG_VIZ_HOUGH_LANE:
            cv2.imshow('hough_lane_edges', edge_img)
            cv2.imshow('hough_lane_result', debug_img)
            cv2.waitKey(1)
        return valid, offset, lookahead, lane_center, path, debug_img


class YellowDashCounter:
    """근거리 ROI에서 노란색 파선(끊긴 중앙선) 조각을 세는 카운터.

    좌회전 진입 트리거 후보 — "신호등→좌회전 입구" 거리를 적분하는 대신, 실제로
    지나간 노란 파선 개수를 센다(config.py "좌회전 진입 랜드마크 후보" 절 참고).
    HSV 마스크(YELLOW_LOWER/UPPER — hough_lane.HoughLaneDetector._update_yellow_centers()와
    동일 기준)로 근거리 밴드 안 노란 픽셀 존재 여부를 매 프레임 판단하고, '있음↔없음'
    전이를 디바운스(연속 프레임 확정 — track_drive.py _update_lap()의
    LAP_YAW_CONFIRM_FRAMES와 동일 패턴)한다.

    카운트는 "다시 안 보이게 되는 순간"(falling edge)에 올라간다 — 아직 눈에 보이기만
    하고 지나치지 않은 파선까지 세면 안 되기 때문. 근거리 ROI 안에서 파선이 사라진다는
    건 차량이 그만큼 전진해 실제로 지나쳤다는 뜻이라, falling edge가 "지나침"에 더
    가깝다.

    판단은 ROI 전체 픽셀수가 아니라 커넥티드컴포넌트 단위로 한다 — 전체 픽셀수만 보면
    트랙 밖 나무색 바닥재가 노란 파선으로 오검출된다(config.py YELLOW_DASH_* 절 주석
    참고). 컴포넌트가 ① 면적이 파선 하나 크기 범위(YELLOW_DASH_MIN/MAX_AREA_PX) 안이고
    ② ROI 좌우 테두리에 안 닿아야(닿으면 ROI 밖에서 잘려 들어온 큰 덩어리의 일부라는
    뜻) "파선 후보"로 인정한다. 가로세로비 필터는 일부러 안 씀 — 화면 가장자리의
    파선은 어안렌즈 왜곡 때문에 대각선으로 찍혀 바운딩박스가 넓적해지므로, 세로가
    길어야 한다는 조건은 오히려 진짜 파선을 놓친다.

    카운팅을 언제부터 시작할지(예: 노란 하프 출발선 검출 시점)는 이 클래스 책임이 아니다
    — 호출부가 reset()을 호출한 시점부터 셀 뿐이다. 상태머신 연결(트리거 시점에 실제로
    조향을 꺾는 것)은 아직 없음 — 우선 카운팅 자체가 실차 캡처에서 파선 하나하나를
    잘 분리해 세는지부터 확인 필요.

    트리거 조건("3개 지나치고 4번째가 다가오는 순간")은 count만으로는 못 만든다 — count는
    "완전히 지나친 개수"라 4번째가 다가오는 중에는 아직 3이다. 호출부는 self.present(현재
    파선이 보이는 중인가, 디바운스 확정값)를 같이 봐서 `count == TURN_DASH_TRIGGER_COUNT-1
    and present`로 판단해야 한다(config.py TURN_DASH_TRIGGER_COUNT 주석 참고).
    """

    def __init__(self):
        self.count = 0
        self.present = False    # 디바운스로 확정된 현재 상태(파선이 보이는 중인가) — 호출부가 직접 읽는 공개 상태
        self._present_run = 0    # raw 신호가 연속으로 "보임"이었던 프레임 수
        self._absent_run = 0     # raw 신호가 연속으로 "안보임"이었던 프레임 수

    def reset(self):
        """카운팅 재시작 — 호출부가 '지금부터 세기 시작'하는 시점에 호출."""
        self.count = 0
        self.present = False
        self._present_run = 0
        self._absent_run = 0

    def update(self, frame):
        """frame(전방 카메라 원본 BGR) 1장 처리 → 갱신된 count 반환.

        frame이 None이면 카운트를 건드리지 않고 현재 값만 반환(카메라 프레임 드롭 때문에
        카운트가 흔들리지 않게 — perc_stopline()의 img_front None 가드와 동일 원칙)."""
        if frame is None:
            return self.count

        h, w = frame.shape[:2]
        y0 = int(h * YELLOW_DASH_ROI_TOP)
        y1 = int(h * YELLOW_DASH_ROI_BOT)
        x0 = int(w * YELLOW_DASH_ROI_X0)
        x1 = int(w * YELLOW_DASH_ROI_X1)
        roi = frame[y0:y1, x0:x1]
        roi_w = roi.shape[1]

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, YELLOW_LOWER, YELLOW_UPPER)
        num, _, stats, _ = cv2.connectedComponentsWithStats(mask)
        boxes = []   # 디버그 표시용 [(x,y,cw,ch,area,accepted)]
        raw_present = False
        for k in range(1, num):
            cx, cy, cw, ch, area = stats[k]
            if area < 30:
                continue
            touches_edge = (cx <= 0) or (cx + cw >= roi_w)
            ok = (YELLOW_DASH_MIN_AREA_PX <= area <= YELLOW_DASH_MAX_AREA_PX) and not touches_edge
            boxes.append((cx, cy, cw, ch, area, ok))
            raw_present = raw_present or ok

        if raw_present:
            self._present_run += 1
            self._absent_run = 0
        else:
            self._absent_run += 1
            self._present_run = 0

        if not self.present and self._present_run >= YELLOW_DASH_PRESENT_FRAMES:
            self.present = True
        elif self.present and self._absent_run >= YELLOW_DASH_ABSENT_FRAMES:
            self.present = False
            self.count += 1   # falling edge = 파선이 근거리 ROI를 벗어남 = 실제로 지나침

        if DEBUG_VIZ_DASH_COUNTER:
            self._draw_debug(roi, boxes, raw_present)

        return self.count

    def _draw_debug(self, roi, boxes, raw_present):
        vis = roi.copy()
        for cx, cy, cw, ch, area, ok in boxes:
            cv2.rectangle(vis, (cx, cy), (cx + cw, cy + ch), GREEN if ok else RED, 2)
            cv2.putText(vis, str(area), (cx, max(cy - 4, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, GREEN if ok else RED, 1, cv2.LINE_AA)
        border = GREEN if self.present else (RED if raw_present else (120, 120, 120))
        cv2.rectangle(vis, (1, 1), (vis.shape[1] - 2, vis.shape[0] - 2), border, 3)
        cv2.putText(vis, f'count={self.count} present={self.present} raw={raw_present}',
                    (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow('yellow_dash_counter', vis)
        cv2.waitKey(1)


class CheckerBandGate:
    """흑/노랑 체크무늬 밴드("하프 출발선") 통과 감지 — YellowDashCounter.reset() 호출
    시점(카운팅 시작 트리거) 후보.

    check_stopline()(perc_floor.py)과 같은 ROI+비율 임계값 방식이지만, 정지선(흰색
    단일톤)과 달리 이 밴드는 "어두운 비율"과 "노란 비율"이 같은 ROI 안에 동시에 있어야
    확정한다(config.py "체크무늬 게이트 밴드" 절 참고) — 그래야 맨 도로나 순수 노란
    점선 중앙선과 안 헷갈린다. CHECKER_CONFIRM_FRAMES 연속 확정되면 딱 한 번만 True를
    반환(rising edge, _update_lap()의 디바운스와 동일 패턴) — reset() 전까지 재무장 안
    되므로, 밴드 위를 지나는 여러 프레임 동안 카운터가 여러 번 리셋되는 일은 없다.
    """

    def __init__(self):
        self._confirm_run = 0
        self._armed = True    # False가 되면(한 번 트리거된 후) reset() 전까지 다시 안 켜짐

    def reset(self):
        """다음 랩/재진입을 위해 재무장 — 호출부(상태머신)가 좌회전 시퀀스를 완전히
        마친 뒤 호출."""
        self._confirm_run = 0
        self._armed = True

    def update(self, frame):
        """frame(전방 카메라 원본 BGR) 1장 처리 → 이번 프레임에 새로 확정됐으면 True."""
        if frame is None or not self._armed:
            return False

        h, w = frame.shape[:2]
        y0 = int(h * CHECKER_ROI_TOP)
        y1 = int(h * CHECKER_ROI_BOT)
        roi = frame[y0:y1, 0:w]

        # "하프" 출발선 — 체커무늬가 도로 폭 전체가 아니라 우측 차로에만 있다. ROI
        # 전체 폭 평균으로 비율을 재면 무늬 없는 좌측 절반이 희석시켜 임계값을 못 넘길
        # 수 있어, 우측 절반만 잘라서 검사한다.
        mid = roi.shape[1] // 2
        right_half = roi[:, mid:]
        dark_ratio, yellow_ratio = self._checker_ratios(right_half)
        raw_present = (dark_ratio >= CHECKER_DARK_RATIO_TH
                       and yellow_ratio >= CHECKER_YELLOW_RATIO_TH)

        self._confirm_run = self._confirm_run + 1 if raw_present else 0
        triggered = self._confirm_run >= CHECKER_CONFIRM_FRAMES
        if triggered:
            self._armed = False

        if DEBUG_VIZ_CHECKER_GATE:
            self._draw_debug(roi, mid, dark_ratio, yellow_ratio, raw_present, triggered)

        return triggered

    @staticmethod
    def _checker_ratios(half_roi):
        gray = cv2.cvtColor(half_roi, cv2.COLOR_BGR2GRAY)
        dark_ratio = float(np.count_nonzero(gray < CHECKER_BLACK_MAX_V)) / gray.size

        hsv = cv2.cvtColor(half_roi, cv2.COLOR_BGR2HSV)
        yellow_mask = cv2.inRange(hsv, YELLOW_LOWER, YELLOW_UPPER)
        yellow_ratio = float(np.count_nonzero(yellow_mask)) / yellow_mask.size

        return dark_ratio, yellow_ratio

    def _draw_debug(self, roi, mid, dark_ratio, yellow_ratio, raw_present, triggered):
        """전체 ROI(좌+우)를 그대로 보여주되, 실제로 검사 대상인 우측 절반만 테두리+반투명
        색으로 하이라이트하고 그 위에 체크(✓)/엑스(x) 표시를 얹는다 — 크롭된 절반만 보여주면
        ROI가 실제 체커무늬 위치와 잘 맞는지 전체 맥락에서 확인하기 어렵다는 지적 반영."""
        vis = roi.copy()
        h, w = vis.shape[:2]
        color = GREEN if raw_present else RED

        overlay = vis.copy()
        cv2.rectangle(overlay, (mid, 0), (w - 1, h - 1), color, -1)
        cv2.addWeighted(overlay, 0.25, vis, 0.75, 0, vis)
        cv2.rectangle(vis, (mid, 0), (w - 1, h - 1), color, 2)
        cv2.line(vis, (mid, 0), (mid, h - 1), (255, 255, 255), 1)  # 좌/우 분할선 — 검사 안 하는 좌측과 경계 표시

        mark_cx, mark_cy = mid + (w - mid) // 2, h // 2
        self._draw_check_mark(vis, (mark_cx, mark_cy), raw_present)

        cv2.putText(vis, f'dark={dark_ratio:.2f}(>={CHECKER_DARK_RATIO_TH}) '
                          f'yel={yellow_ratio:.2f}(>={CHECKER_YELLOW_RATIO_TH})',
                    (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(vis, f'confirm={self._confirm_run}/{CHECKER_CONFIRM_FRAMES} '
                          f'armed={self._armed}',
                    (6, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        if triggered:
            cv2.putText(vis, 'TRIGGERED', (6, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow('checker_band_gate', vis)
        cv2.waitKey(1)

    @staticmethod
    def _draw_check_mark(img, center, ok, size=22, thickness=3):
        cx, cy = center
        color = GREEN if ok else RED
        if ok:
            cv2.line(img, (cx - size, cy), (cx - size // 3, cy + size), color, thickness, cv2.LINE_AA)
            cv2.line(img, (cx - size // 3, cy + size), (cx + size, cy - size), color, thickness, cv2.LINE_AA)
        else:
            cv2.line(img, (cx - size, cy - size), (cx + size, cy + size), color, thickness, cv2.LINE_AA)
            cv2.line(img, (cx - size, cy + size), (cx + size, cy - size), color, thickness, cv2.LINE_AA)
