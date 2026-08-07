#=============================================
# vesc_twist_bridge.py — '/vesc_speed_erpm'(std_msgs/Float32, ERPM)을
# robot_localization EKF(config/ekf.yaml, §9)가 바로 융합할 수 있는
# geometry_msgs/TwistWithCovarianceStamped로 재발행하는 ROS2 노드.
#
# ── 왜 필요한가 ──
#   track_drive.py의 cb_vesc()는 이미 self.v_mps로 변환해 LQR 게인/pose_estimator에
#   쓰고 있지만(config.py "VESC 실측 속도 연동" 절 참고), 그건 track_drive 노드
#   내부 상태일 뿐 다른 노드가 구독할 수 있는 토픽이 아니다. slam_toolbox 매핑
#   (§9)과 AMCL 실주행 로컬라이제이션이 쓰는 robot_localization EKF는 별도
#   프로세스라 독립된 토픽이 필요해서 이 작은 브리지를 하나 더 둔다 — track_drive.py의
#   cb_vesc()/VESC_SPEED_TO_ERPM_GAIN과 정확히 같은 환산식을 그대로 재사용한다.
#
# ── 커버리언스 값 관련 ──
#   linear.x 분산은 VESC 홀센서 자체의 노이즈를 실측하지 않아 임의값(0.01, ≈±0.1m/s
#   1시그마)이다 — EKF 융합 결과가 이상하면 여기부터 실측/튜닝할 것. 다른 축(y, z,
#   각속도)은 이 로봇이 융합에 안 쓰므로(config/ekf.yaml의 twist0_config 참고) 값
#   자체는 의미 없지만, REP-105 관례대로 큰 분산(신뢰 안 함)으로 채워둔다.
#
# ── 죽었을 때 동작 ──
#   /vesc_speed_erpm 메시지를 받을 때만 재발행한다(주기적 타이머로 마지막 값을
#   반복 발행하지 않음) — VESC 브리지(launch/vesc_speed_bridge.py, ROS1)가 죽으면
#   이 노드도 조용히 발행을 멈추고, EKF는 그 시간 동안 IMU만으로 예측 단계를
#   진행한다(config.py의 "센서 죽음은 조용한 고정값으로 나타남" 패턴과 동일하게,
#   에러 없이 "값이 안 들어옴"으로만 드러나니 EKF 튜닝 시 참고).
#=============================================
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
from geometry_msgs.msg import TwistWithCovarianceStamped
from rclpy.qos import qos_profile_sensor_data

from ..config import VESC_SPEED_TO_ERPM_GAIN

# REP-105 관례: 융합 안 하는 축은 크게, 융합하는 축(linear.x)만 실측 전 임의 추정치.
_UNUSED_VAR = 1e3
_LINEAR_X_VAR = 0.01


class VescTwistBridge(Node):

    def __init__(self):
        super().__init__('vesc_twist_bridge')
        self.pub = self.create_publisher(TwistWithCovarianceStamped, '/vesc/twist_raw', 10)
        self.create_subscription(
            Float32, '/vesc_speed_erpm', self.cb_vesc, qos_profile_sensor_data)

    def cb_vesc(self, msg):
        v_mps = float(msg.data) / VESC_SPEED_TO_ERPM_GAIN

        out = TwistWithCovarianceStamped()
        out.header.stamp = self.get_clock().now().to_msg()
        out.header.frame_id = 'base_link'
        out.twist.twist.linear.x = v_mps

        cov = [_UNUSED_VAR] * 36
        cov[0] = _LINEAR_X_VAR  # linear.x
        out.twist.covariance = cov

        self.pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = VescTwistBridge()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
