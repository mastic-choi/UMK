#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================
# color_picker_node.py — 차선 색상 캘리브레이션 전용 노드
#
# track_drive 노드와 별도로 띄워서, lane_util.CameraProcessor가 만드는 것과 동일한
# BEV 화면을 보여주고 마우스로 클릭하면 그 픽셀의 HSV 값을 뽑아 누적 통계(min~max)를
# 터미널에 출력한다. lane_util.py/track_drive.py는 전혀 건드리지 않는다.
#
# 실행:
#   ros2 run track_drive color_picker   (setup.py entry_points에 등록한 경우)
#   또는 python3 color_picker_node.py   (ROS2 환경이 소싱된 상태에서 직접 실행)
#
# 사용법:
#   lane_bev 창에서 차선 위 여러 지점을 클릭 → 터미널에 HSV 및 누적 범위 출력
#   'c' 키: 누적 샘플 초기화
#   'q' 키: 노드 종료
# =============================================
import rclpy
import cv2
import numpy as np
from rclpy.node import Node
from sensor_msgs.msg import Image
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from cv_bridge import CvBridge

from .lane_util import CameraProcessor

WINDOW_NAME = 'color_picker_bev'


class ColorPickerNode(Node):
    def __init__(self):
        super().__init__('color_picker_node')

        self.bridge = CvBridge()
        self.camera_processor = CameraProcessor()
        self.frame = None
        self.color_samples = []   # 클릭으로 모은 (H, S, V) 샘플들
        self._mouse_bound = False

        image_qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                                history=HistoryPolicy.KEEP_LAST, depth=1)
        self.create_subscription(Image, '/usb_cam/image_raw/front',
                                  self.cb_img_front, image_qos)

        self.get_logger().info(
            f"색상 캘리브레이션 노드 시작. '{WINDOW_NAME}' 창에서 차선을 클릭하세요. "
            "('c'=초기화, 'q'=종료)"
        )

    def cb_img_front(self, msg):
        try:
            self.frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            self.get_logger().warn(f'이미지 변환 실패: {e}')
            return

        # lane_util.CameraProcessor가 실제 주행에 쓰는 것과 동일한 BEV/마스크 생성 로직 재사용
        bev, white, yellow = self.camera_processor.processor(self.frame)
        if bev is None:
            return

        cv2.imshow(WINDOW_NAME, bev)
        if not self._mouse_bound:
            cv2.setMouseCallback(WINDOW_NAME, self._on_click)
            self._mouse_bound = True

        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            self.color_samples = []
            self.get_logger().info('[색상샘플] 초기화')
        elif key == ord('q'):
            self.get_logger().info('종료 요청 수신')
            rclpy.shutdown()

    def _on_click(self, event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        bev = self.camera_processor.bev
        if bev is None or not (0 <= y < bev.shape[0] and 0 <= x < bev.shape[1]):
            return

        bgr = bev[y, x]
        h, s, v = cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2HSV)[0][0]
        self.color_samples.append((int(h), int(s), int(v)))

        samples = np.array(self.color_samples)
        lo = samples.min(axis=0)
        hi = samples.max(axis=0)
        self.get_logger().info(
            f'({x},{y}) BGR={tuple(int(c) for c in bgr)} HSV=({h},{s},{v}) '
            f'| n={len(self.color_samples)} 범위 H:{lo[0]}-{hi[0]} S:{lo[1]}-{hi[1]} V:{lo[2]}-{hi[2]}'
        )


def main(args=None):
    rclpy.init(args=args)
    node = ColorPickerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
