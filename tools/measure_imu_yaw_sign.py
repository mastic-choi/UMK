#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#=============================================
# measure_imu_yaw_sign.py — IMU 요레이트(angular_velocity.z) 부호규약 실측 도구.
#
# ── 왜 필요한가 ──
# README §0.5.5: "IMU 각속도 부호규약(z축 +가 좌/우 중 어느 쪽인지)이 실차
# 미검증"이라고 명시돼 있다. 지금까지는 _imu_curvature_px() 등에서 abs()로만
# 써서(부호를 안 씀) 문제가 안 됐지만, local_costmap_proposal.md의 자기위치
# 추정(그리드를 "차가 회전한 방향으로" 같이 돌리는 것)에는 부호가 그대로 필요하다 —
# 차를 실제로 알려진 방향으로 돌려보고 그때 z값 부호를 재는 것 외엔 확인할 방법이
# 없다.
#
# ── 원리 ──
# 차량(또는 IMU 보드 그 자체)을 손으로 들어, 위에서 내려다봤을 때 반시계(왼쪽)/
# 시계(오른쪽) 방향으로 천천히 돌리면서 angular_velocity.z 평균을 잰다. 왼쪽↔오른쪽을
# 번갈아 여러 번 반복해서, 매번 부호가 방향과 일관되게 뒤집히는지 확인한다(우연히
# 한 번 맞은 게 아니라는 근거).
#
# ── 이 파일은 ROS2 빌드/노드 없이 그냥 실행합니다 ──
#   python3 measure_imu_yaw_sign.py
# rclpy + IMU 노드(/imu 발행)만 있으면 된다 — track_drive 전체 launch 불필요,
# track_drive 패키지도 import하지 않는다.
#
# ── 사용법 ──
#   1) IMU 노드만 띄운다(xycar_imu 등, track_drive 전체 launch 불필요).
#   2) python3 measure_imu_yaw_sign.py [--trials 4] [--duration 2.5] 실행.
#   3) 안내에 따라 차량(또는 IMU 보드)을 왼쪽/오른쪽으로 번갈아 천천히 돌리며 진행
#      (트랙 위 실차가 아니라 손으로 들고 해도 됨 — IMU 자체 회전만 재면 됨).
#   4) 마지막에 "+z = 좌회전" 또는 "+z = 우회전" 결론과, README/config.py에 남길
#      문구를 출력한다.
#=============================================
import argparse
import sys
import time


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--imu-topic', default='/imu')
    ap.add_argument('--trials', type=int, default=4,
                     help='왼쪽/오른쪽 번갈아 반복할 횟수(짝수 권장, 기본 4=왼2+오2)')
    ap.add_argument('--duration', type=float, default=2.5, help='회차당 회전+측정 시간(초)')
    args = ap.parse_args()

    try:
        import rclpy
        from rclpy.node import Node
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import Imu
    except ImportError:
        print('rclpy가 없습니다 — 이 도구는 실차(ROS2 환경)에서만 실행할 수 있습니다.')
        return 1

    # (timestamp, z) 로그 하나만 계속 쌓는다 — 콜백을 회차마다 바꿔치는 대신, 각
    # 회차의 시작/끝 시각으로 로그를 나중에 잘라서 쓴다(콜백 재할당보다 훨씬 안전).
    log = []

    class _Probe(Node):
        def __init__(self):
            super().__init__('measure_imu_yaw_sign_probe')
            self.sub = self.create_subscription(
                Imu, args.imu_topic, self.cb, qos_profile_sensor_data)

        def cb(self, msg):
            log.append((time.time(), msg.angular_velocity.z))

    print(f'=== IMU 요레이트(angular_velocity.z) 부호규약 실측 — {args.trials}회 ===')
    print(f'토픽: {args.imu_topic}. 매 회차 {args.duration:.1f}초간 재는 동안 안내된 방향으로')
    print('천천히(부드럽게, 급격하지 않게) 계속 회전시켜 주세요. 위에서 내려다본 기준입니다.\n')

    rclpy.init(args=[])
    node = _Probe()

    left_samples = []   # (반시계) 회차 평균들
    right_samples = []  # (시계) 회차 평균들
    try:
        for i in range(args.trials):
            direction = '왼쪽(반시계)' if i % 2 == 0 else '오른쪽(시계)'
            input(f'--- 회차 {i+1}/{args.trials}: [{direction}]로 돌릴 준비가 되면 Enter ---')
            print(f'  지금부터 {args.duration:.1f}초간 {direction}로 천천히 돌리세요...')

            t0 = time.time()
            while time.time() - t0 < args.duration:
                rclpy.spin_once(node, timeout_sec=0.1)
            t1 = time.time()

            samples = [z for (t, z) in log if t0 <= t <= t1]
            if not samples:
                print('  (이 회차엔 메시지를 못 받았습니다 — 건너뜁니다)')
                continue
            mean_z = sum(samples) / len(samples)
            print(f'  회차 평균 z = {mean_z:+.4f} rad/s ({len(samples)}개 샘플)\n')
            (left_samples if i % 2 == 0 else right_samples).append(mean_z)
    finally:
        node.destroy_node()
        rclpy.shutdown()

    if not left_samples or not right_samples:
        print('왼쪽/오른쪽 둘 다 최소 1회씩은 측정돼야 결론을 낼 수 있습니다 — '
              '샘플이 부족해 종료합니다.')
        return 1

    left_avg = sum(left_samples) / len(left_samples)
    right_avg = sum(right_samples) / len(right_samples)
    left_consistent = all((s > 0) == (left_avg > 0) for s in left_samples)
    right_consistent = all((s > 0) == (right_avg > 0) for s in right_samples)
    opposite_signs = (left_avg > 0) != (right_avg > 0)

    print('=== 결과 ===')
    print(f'왼쪽(반시계) 회차 평균들: {[f"{s:+.3f}" for s in left_samples]} → 전체평균 {left_avg:+.4f}')
    print(f'오른쪽(시계) 회차 평균들: {[f"{s:+.3f}" for s in right_samples]} → 전체평균 {right_avg:+.4f}')

    if not (left_consistent and right_consistent and opposite_signs):
        print('\n★주의★ 방향과 부호가 일관되게 안 갈렸습니다 — 회전이 너무 약했거나')
        print('  (IMU가 거의 안 움직였거나) 좌/우를 바꿔 돌리다 헷갈렸을 수 있습니다.')
        print('  --duration을 늘리고 더 확실하게 돌려서 다시 시도해보세요.')
        return 1

    sign_desc = '좌회전(반시계)' if left_avg > 0 else '우회전(시계)'
    print(f'\nOK — 왼쪽/오른쪽에서 부호가 일관되게 반대로 나왔습니다.')
    print(f'결론: angular_velocity.z 양수(+) = {sign_desc}')

    print('\nREADME §0.5.5 "알려진 한계" / config.py IMU 관련 주석에 남길 문구:')
    print(f'  # [실측 2026-08-17] IMU angular_velocity.z 부호규약 확인 완료 —')
    print(f'  #   z > 0 = {sign_desc}. (measure_imu_yaw_sign.py로 좌/우 번갈아')
    print(f'  #   {len(left_samples)+len(right_samples)}회 실측, 부호 일관성 확인됨)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
