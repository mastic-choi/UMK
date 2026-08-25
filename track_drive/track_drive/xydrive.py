#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================
# xydrive.py — xymotor(모터 드라이버) 단독 실행 테스트용 최소 노드
#
# track_drive.py 전체를 켜지 않고도 'xycar_motor' 토픽에 angle=0(직진), 고정 speed를
# 계속 발행만 해서 xymotor 노드가 실제로 바퀴를 굴리는지만 확인하기 위한 용도.
# 발행 형식(Float32MultiArray [angle, speed], 주기마다 7회 반복 발행)은
# track_drive.py의 drive()를 그대로 따른다 — 실차 모터 드라이버가 이 형식만 받는 게
# 이미 검증되어 있기 때문(track_drive.py 참고).
#
# 실행:
#   ros2 run track_drive xydrive
#   ros2 run track_drive xydrive --ros-args -p speed:=5.0
# =============================================
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray

DEFAULT_SPEED = 5.0  # 테스트용 저속 — 필요하면 speed 파라미터로 조절


class XyDrive(Node):
    def __init__(self):
        super().__init__('xydrive')
        self.declare_parameter('speed', DEFAULT_SPEED)
        self.speed = float(self.get_parameter('speed').value)

        self.motor_msg = Float32MultiArray()
        self.motor_pub = self.create_publisher(Float32MultiArray, 'xycar_motor', 10)
        self.create_timer(0.05, self.control_loop)

        self.get_logger().info(f'xydrive 시작 | 직진 speed={self.speed}')

    def control_loop(self):
        self.motor_msg.data = [0.0, self.speed]
        for _ in range(7):
            self.motor_pub.publish(self.motor_msg)


def main(args=None):
    rclpy.init(args=args)
    node = XyDrive()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.motor_msg.data = [0.0, 0.0]
        for _ in range(7):
            node.motor_pub.publish(node.motor_msg)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
