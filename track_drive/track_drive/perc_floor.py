import cv2
import numpy as np

# ── 정지선 감지 튜닝 파라미터 (전전년도 팀 실측값 그대로, 카메라 해상도 640x480 고정 가정) ──
#   프로젝트 전체가 640x480 한 대의 카메라를 전제로 캘리브레이션돼 있어(BEV_SRC/DST, SIG_ROI 등)
#   해상도가 달라지면 이 값들 전부 재보정 대상이므로, 비율 변환 없이 절대픽셀 그대로 사용.
#   튜닝도 디버그 창에서 눈으로 사각형 맞추는 방식이라 절대픽셀이 더 직관적.
STOPLINE_ROI_Y0, STOPLINE_ROI_Y1 = 270, 320   # 세로 밴드
STOPLINE_ROI_X0, STOPLINE_ROI_X1 = 150, 480   # 가로 중앙 크롭
STOPLINE_WHITE_LOW = 180                      # 그레이스케일 흰색 임계
STOPLINE_TH = 0.06                            # ROI 내 흰 픽셀 비율 임계 (실측: 1000/16500 ≈ 6%)
DEBUG_VIZ_STOPLINE = False


def check_stopline(image):
    """
    굵은 가로 흰선(정지선) 감지 → True/False 반환.
      - 입력 : 전방 카메라 BGR 이미지 (640x480 가정)
      - 출력 : 정지선 감지 여부 (bool)
    """
    if image is None:
        return False

    roi = image[STOPLINE_ROI_Y0:STOPLINE_ROI_Y1, STOPLINE_ROI_X0:STOPLINE_ROI_X1]

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, STOPLINE_WHITE_LOW, 255, cv2.THRESH_BINARY)

    white_ratio = float(np.count_nonzero(binary)) / binary.size
    detected = white_ratio > STOPLINE_TH

    if DEBUG_VIZ_STOPLINE:
        vis = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        cv2.putText(vis, f'ratio={white_ratio:.3f} th={STOPLINE_TH:.2f}',
                    (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(vis, 'DETECTED' if detected else 'none',
                    (4, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 255) if detected else (200, 200, 200), 1, cv2.LINE_AA)
        cv2.imshow('stopline', vis)
        cv2.waitKey(1)

    return detected


class LaneDetector:
    """CameraProcessor(BEV/마스크) + SlideWindow(차선 탐색/피팅)을 묶은 최종 통합 인식기."""

    def __init__(self, camera_processor=None, slide_window_processor=None):
        self.camera_processor = camera_processor
        self.slide_window_processor = slide_window_processor

    def set_processor(self, camera, slide_window):
        self.camera_processor = camera
        self.slide_window_processor = slide_window

    def detect(self, frame):
        """
        입력 : 전방 카메라 BGR 프레임
        출력 : (lane_valid, lane_offset, lane_lookahead, lane_center, bev)
        """
        bev, yellow_mask = self.camera_processor.processor(frame)

        if bev is None:
            return False, 0.0, 0.0, 320.0, None   # lane_center는 화면 중앙(640/2)을 기본값으로

        lane_valid, lane_offset, lookahead, lane_center = self.slide_window_processor.detect(
            bev, yellow_mask
        )

        return lane_valid, lane_offset, lookahead, lane_center, bev
