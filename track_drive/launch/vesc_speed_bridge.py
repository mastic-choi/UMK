#!/usr/bin/env python
# -*- coding: utf-8 -*-
#=============================================
# vesc_speed_bridge.py — ROS1 노드.
#
# VESC 드라이버가 /sensors/core(vesc_msgs/VescStateStamped)로 발행하는 상태 메시지
# 중 회전속도(state.speed, ERPM) 하나만 뽑아서 std_msgs/Float32로 '/vesc_speed_erpm'에
# 다시 뿌린다.
#
# 왜 필요한가:
#   ROS2쪽(track_drive 워크스페이스)에 vesc_msgs 패키지가 안 빌드돼 있어서(2026-08-06
#   실차 확인) ros1_bridge가 /sensors/core(커스텀 메시지)를 ROS2로 못 넘긴다.
#   ros1_bridge의 동적 브리지는 양쪽에 같은 메시지 패키지가 다 있어야 커스텀 타입을
#   넘길 수 있는데, std_msgs는 ROS1/ROS2 둘 다 기본 내장이라 이 값 하나만 표준 메시지로
#   다시 뿌리면 별도 빌드 없이 ros1_bridge가 자동으로 브리지해준다.
#
# 실행 방법:
#   1) 이 파일을 noetic_ws 안 아무 기존 패키지의 scripts/ 폴더에 넣거나(예:
#      noetic_ws/src/xycar_motor/scripts/vesc_speed_bridge.py), 새 패키지를 만들어도 됨.
#   2) chmod +x vesc_speed_bridge.py
#   3) ROS1 환경 소싱 후 실행: rosrun <패키지명> vesc_speed_bridge.py
#      (혹은 그냥 python3 vesc_speed_bridge.py 로 직접 실행해도 동작함 — catkin 빌드
#      없이도 rospy 스크립트는 바로 돌아간다. 다만 매번 수동 실행은 번거로우니 기존에
#      xycar_motor.py/vesc_driver를 띄우는 launch 파일에 노드로 하나 추가해두는 걸 권장)
#
# 확인 방법:
#   ROS2쪽에서 track_drive 노드를 띄운 상태로 'vesc_debug' 디버그 창이 초록(LIVE)으로
#   바뀌는지 확인. 또는 ROS1쪽에서 `rostopic echo /vesc_speed_erpm`으로 값이 찍히는지,
#   ROS2쪽에서 `ros2 topic echo /vesc_speed_erpm`으로 넘어오는지 직접 확인해도 된다.
#=============================================
import rospy
from std_msgs.msg import Float32
from vesc_msgs.msg import VescStateStamped

_pub = None


def cb(msg):
    _pub.publish(Float32(data=msg.state.speed))


def main():
    global _pub
    rospy.init_node('vesc_speed_bridge')
    _pub = rospy.Publisher('/vesc_speed_erpm', Float32, queue_size=1)
    rospy.Subscriber('/sensors/core', VescStateStamped, cb, queue_size=1)
    rospy.loginfo('[vesc_speed_bridge] /sensors/core -> /vesc_speed_erpm(Float32) 변환 시작')
    rospy.spin()


if __name__ == '__main__':
    main()
