#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================
# manual_drive_collector.py — 수동 조향 주행 + 재생 테스트용 원본 영상 수집 노드
#
# 터틀심 teleop처럼 터미널에서 w/a/s/d로 직접 조향/속도를 주면서 주행하고,
# 원본(ROI/BEV 적용 전) 카메라 프레임을 동영상으로, 그 시점의 angle/speed/IMU yaw를
# meta.jsonl로 기록한다.
#
# BEV는 저장하지 않는다 — CameraProcessor.processor(raw_frame)가 ROI+BEV+마스크를
# 그대로 재현하므로, raw 영상만 있으면 나중에 인식 코드(HSV든 세그멘테이션 모델이든)에
# 그대로 흘려보내 재생 테스트할 수 있다. BEV_SRC/DST 캘리브레이션이 나중에 바뀌어도
# raw 영상은 항상 최신 기준으로 다시 처리되므로 더 안전하다(사전 계산된 BEV를 저장하면
# 캘리브레이션이 바뀌는 순간 낡은 데이터가 되어버림).
#
# 휠 오도메트리는 이 워크스페이스에 아예 없어서(xycar_msgs에 관련 메시지 타입 자체가 없음)
# 대신 우리가 실제로 내보낸 명령값(angle, speed)을 시계열로 남긴다.
#
# track_drive.py/lane_util.py는 건드리지 않는다. 모터 발행 형식(Float32MultiArray
# [angle, speed], 7회 반복 발행)은 track_drive.py의 drive()를 그대로 따른다 — 실차
# 모터 드라이버가 이 형식만 받는 게 이미 검증되어 있기 때문.
#
# 실행:
#   ros2 run track_drive manual_drive_collector
#   또는 python3 -m track_drive.manual_drive_collector --out_dir ./lane_seg/data/manual_drive --init_speed 8
#
# 조작:
#   w/s : 속도 +/- (SPEED_STEP)      a/d : 조향 좌/우(ANGLE_STEP)
#   x   : 조향 0으로 리셋(속도 유지)   space : 완전 정지(속도 0, 조향 0)
#   q   : 종료
#
# 저장 구조 (실행할 때마다 새 세션 폴더 생성):
#   out_dir/session_YYYYmmdd_HHMMSS/
#     raw.mp4      원본 카메라 프레임 동영상 (프레임 순서 = meta.jsonl의 idx 순서)
#     meta.jsonl   한 줄에 하나씩: {idx, t, angle, speed, imu_yaw}
# =============================================
import argparse
import json
import math
import select
import sys
import termios
import time
import tty
from pathlib import Path

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, Imu
from std_msgs.msg import Float32MultiArray
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, qos_profile_sensor_data
from cv_bridge import CvBridge

from .lane_util import CameraProcessor

ANGLE_MAX = 100.0
ANGLE_STEP = 5.0
SPEED_STEP = 1.0
SPEED_CLAMP = 100.0

WINDOW_NAME = 'manual_drive_raw'


def get_key(settings, timeout=0.0):
    """터미널을 raw 모드로 바꿔 논블로킹(timeout초)으로 키 하나를 읽는다.
    ROS2 teleop_twist_keyboard와 동일한 표준 패턴."""
    tty.setraw(sys.stdin.fileno())
    rlist, _, _ = select.select([sys.stdin], [], [], timeout)
    key = sys.stdin.read(1) if rlist else ''
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    return key


class ManualDriveCollector(Node):
    def __init__(self, out_dir: Path, init_speed: float, save_hz: float):
        super().__init__('manual_drive_collector')

        session_name = time.strftime('session_%Y%m%d_%H%M%S')
        self.session_dir = out_dir / session_name
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.video_path = self.session_dir / 'raw.mp4'
        self.meta_path = self.session_dir / 'meta.jsonl'
        self._meta_file = open(self.meta_path, 'a')
        self._writer = None   # 첫 프레임 크기를 봐야 열 수 있어서 지연 초기화

        self.bridge = CvBridge()
        self.camera_processor = CameraProcessor()   # 라이브 미리보기(BEV)용, 저장은 안 함

        self.angle = 0.0
        self.speed = float(init_speed)
        self.imu_yaw = None
        self._frame_idx = 0
        self.save_hz = save_hz
        self._save_interval = (1.0 / save_hz) if save_hz > 0 else 0.0
        self._last_save_t = 0.0

        self.motor_msg = Float32MultiArray()
        self.motor_pub = self.create_publisher(Float32MultiArray, 'xycar_motor', 10)

        image_qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                                history=HistoryPolicy.KEEP_LAST, depth=1)
        self.create_subscription(Image, '/usb_cam/image_raw/front', self.cb_img_front, image_qos)
        self.create_subscription(Imu, '/imu', self.cb_imu, qos_profile_sensor_data)

        self.create_timer(0.05, self.control_loop)   # 20Hz, track_drive.py와 동일 주기

        self._term_settings = termios.tcgetattr(sys.stdin)

        self.get_logger().info(
            f'세션 폴더: {self.session_dir}\n'
            f'초기 속도={self.speed}, 조작: w/s=속도 a/d=조향 x=조향리셋 space=정지 q=종료'
        )

    def cb_imu(self, msg):
        q = msg.orientation
        self.imu_yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))

    def cb_img_front(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            self.get_logger().warn(f'이미지 변환 실패: {e}')
            return

        # 라이브 미리보기: 운전 중 지금 차선이 어떻게 보이는지 확인용(디스크 저장 안 함)
        bev, _white, _yellow = self.camera_processor.processor(frame)
        vis = frame.copy()
        cv2.putText(vis, f'angle={self.angle:+.1f} speed={self.speed:+.1f} saved={self._frame_idx}',
                    (4, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow(WINDOW_NAME, vis)
        if bev is not None:
            cv2.imshow('manual_drive_bev', bev)
        cv2.waitKey(1)

        now = time.time()
        if now - self._last_save_t < self._save_interval:
            return
        self._last_save_t = now

        if self._writer is None:
            h, w = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = self.save_hz if self.save_hz > 0 else 20.0
            self._writer = cv2.VideoWriter(str(self.video_path), fourcc, fps, (w, h))

        self._writer.write(frame)
        record = {
            'idx': self._frame_idx,
            't': now,
            'angle': self.angle,
            'speed': self.speed,
            'imu_yaw': self.imu_yaw,
        }
        self._meta_file.write(json.dumps(record) + '\n')
        self._meta_file.flush()
        self._frame_idx += 1

    def control_loop(self):
        key = get_key(self._term_settings, timeout=0.0)
        if key:
            if key == 'w':
                self.speed = min(SPEED_CLAMP, self.speed + SPEED_STEP)
            elif key == 's':
                self.speed = max(-SPEED_CLAMP, self.speed - SPEED_STEP)
            elif key == 'a':
                self.angle = max(-ANGLE_MAX, self.angle - ANGLE_STEP)
            elif key == 'd':
                self.angle = min(ANGLE_MAX, self.angle + ANGLE_STEP)
            elif key == 'x':
                self.angle = 0.0
            elif key == ' ':
                self.angle, self.speed = 0.0, 0.0
            elif key == 'q' or key == '\x03':   # q 또는 Ctrl+C
                self.get_logger().info(f'종료. 총 {self._frame_idx}프레임 저장: {self.session_dir}')
                self._close_outputs()
                rclpy.shutdown()
                return

        self._drive(self.angle, self.speed)

    def _drive(self, angle, speed):
        """track_drive.py의 drive()와 동일한 발행 방식(형식/반복횟수)을 따른다."""
        clipped_angle = float(np.clip(angle, -ANGLE_MAX, ANGLE_MAX))
        clipped_speed = float(np.clip(speed, -SPEED_CLAMP, SPEED_CLAMP))
        self.motor_msg.data = [clipped_angle, clipped_speed]
        for _ in range(7):
            self.motor_pub.publish(self.motor_msg)

    def _close_outputs(self):
        if self._writer is not None:
            self._writer.release()
        if not self._meta_file.closed:
            self._meta_file.close()


def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', default='./lane_seg/data/manual_drive')
    parser.add_argument('--init_speed', type=float, default=5.0)
    parser.add_argument('--save_hz', type=float, default=10.0,
                         help='초당 저장 프레임 수 (0이면 카메라 콜백마다 매번 저장)')
    parsed, ros_args = parser.parse_known_args(args=sys.argv[1:])

    rclpy.init(args=ros_args)
    node = ManualDriveCollector(Path(parsed.out_dir), parsed.init_speed, parsed.save_hz)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, node._term_settings)
        cv2.destroyAllWindows()
        node._close_outputs()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
