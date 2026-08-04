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

# ROI(세로 구간) — app_hough_drive.py 원본의 ROI_START_ROW=300/ROI_END_ROW=380
# (640x480 카메라 기준 절대픽셀)을 비율로 환산해 그대로 이식.
#   ※ app_hough_drive.py는 xycar_cam, track_drive는 usb_cam(전방)으로 카메라/마운트가
#     달라 이 ROI가 트랙 노면을 그대로 담는지는 실차에서 재확인 필요.
HOUGH_ROI_TOP = 300 / 480
HOUGH_ROI_BOT = 380 / 480

# Canny / HoughLinesP 파라미터 — app_hough_drive.py 원본 값 그대로.
GAUSSIAN_KSIZE = 5
CANNY_LOW = 60
CANNY_HIGH = 75
HOUGH_RHO = 1
HOUGH_THETA = math.pi / 180
HOUGH_THRESHOLD = 50
HOUGH_MIN_LINE_LEN = 50
HOUGH_MAX_LINE_GAP = 20

# 기울기 절대값이 이 값 이하면 수평선(차선 아님)으로 보고 제외 — 원본과 동일.
SLOPE_MIN = 0.2

# 기준행 위치 (ROI 세로비율, 0=원거리/ROI 위쪽 ~ 1=근거리/ROI 아래쪽).
#   원본은 L_ROW=40/ROI_HEIGHT=80 로 정확히 중간(0.5) 한 지점만 썼다.
#   여기서는 near/far 두 지점으로 나누되, ROI 경계에 너무 붙으면(외삽 오차 커짐)
#   Hough 선분이 짧아 대표직선이 불안정해지므로 가장자리에서 조금 띄웠다.
NEAR_ROW_RATIO = 0.80
FAR_ROW_RATIO = 0.20

# 한쪽 차선만 검출됐을 때 반대쪽 위치 추정에 쓰는 도로폭(px) 초기값 및 그 후 EMA 갱신.
#   원본 app_hough_drive.py의 "x_left = x_right - 380"/"x_right = x_left + 380" 그대로.
#   실차 재검증 필요(카메라 마운트가 다르면 이 폭도 다시 재봐야 함).
DEFAULT_LANE_WIDTH = 380.0
LANE_WIDTH_MIN, LANE_WIDTH_MAX = 180.0, 400.0
LANE_WIDTH_EMA_ALPHA = 0.1

# 노란 중앙선 위치 추정용 색상 마스크 — Hough 로직 자체는 색상을 쓰지 않지만,
# obstacle_avoidance의 회피방향 판단(choose_side)이 lane_side(내가 어느 차선에
# 있는지)에 의존하므로, 그 판정에 필요한 최소한의 정보만 별도로 뽑아 제공한다.
YELLOW_LOWER = np.array([15, 80, 80])
YELLOW_UPPER = np.array([40, 255, 255])

RED = (0, 0, 255)
YELLOW_COL = (0, 255, 255)
BLUE = (255, 0, 0)
GREEN = (0, 255, 0)

DEBUG_VIZ_HOUGH_LANE = True


class HoughLaneDetector:
    """
    기존 LaneDetector(BEV+moments, perc_floor.py)와 동일한
        detect(frame) -> (lane_valid, lane_offset, lane_lookahead, lane_center, debug_img)
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
            return False, 0.0, 0.0, lane_center, None

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

        # 기울기 부호(음수=좌, 양수=우) + 화면 위치로 좌/우 선분 분류 (원본과 동일)
        left_lines, right_lines = [], []
        for line in all_lines:
            x1, ly1, x2, ly2 = line[0]
            slope = 1000.0 if x2 == x1 else float(ly2 - ly1) / float(x2 - x1)
            if abs(slope) <= SLOPE_MIN:
                continue
            if slope < 0 and x2 < self.roi_w / 2.0:
                left_lines.append((x1, ly1, x2, ly2))
                cv2.line(debug_img, (x1, ly1), (x2, ly2), RED, 2)
            elif slope > 0 and x1 > self.roi_w / 2.0:
                right_lines.append((x1, ly1, x2, ly2))
                cv2.line(debug_img, (x1, ly1), (x2, ly2), YELLOW_COL, 2)

        if not left_lines and not right_lines:
            return self._finish(False, 0.0, 0.0, debug_img, edge_img)

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

        if DEBUG_VIZ_HOUGH_LANE:
            self._draw_debug(debug_img, near_row, x_left_near, x_right_near, near_center)

        return self._finish(True, offset, lookahead, debug_img, edge_img, lane_center)

    def _fit_line(self, lines):
        """선분들의 기울기·양끝점 평균으로 대표직선 (m, b)를 구한다. m==0(못 찾음)이면 None."""
        if not lines:
            return None
        x_sum = y_sum = m_sum = 0.0
        for x1, y1, x2, y2 in lines:
            x_sum += x1 + x2
            y_sum += y1 + y2
            m_sum += (float(y2 - y1) / float(x2 - x1)) if x2 != x1 else 0.0
        size = len(lines)
        m = m_sum / size
        if m == 0.0:
            return None
        x_avg, y_avg = x_sum / (size * 2), y_sum / (size * 2)
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

    def _finish(self, valid, offset, lookahead, debug_img, edge_img, lane_center=None):
        if lane_center is None:
            lane_center = self.roi_w / 2.0 if self.roi_w else 0.0
        if DEBUG_VIZ_HOUGH_LANE:
            cv2.imshow('hough_lane_edges', edge_img)
            cv2.imshow('hough_lane_result', debug_img)
            cv2.waitKey(1)
        return valid, offset, lookahead, lane_center, debug_img
