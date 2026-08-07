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

# ROI(세로 구간) — 한 차례 200~400px(220px 밴드)로 넓혔던 것을, 중심(≈300px)은
# 유지한 채 밴드 높이만 1/3(≈73px)로 되돌렸다. app_hough_drive.py 원본(300~380,
# 80px 밴드)과 거의 같은 폭.
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
from ..config import YELLOW_LOWER, YELLOW_UPPER, DEBUG_VIZ_HOUGH_LANE

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
