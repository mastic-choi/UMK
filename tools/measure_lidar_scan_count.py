#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#=============================================
# measure_lidar_scan_count.py — 라이다 한 스캔당 포인트 개수(n) 실측 도구.
#
# ── 왜 필요한가 ──
# perc_obstacle()/perc_lavacon_trigger()(track_drive.py)의 라이다 인덱스->각도 변환이
# 아직 "m = min(len(ranges), 360)"으로 n=360을 가정한 옛 공식을 쓴다:
#   deg = linspace(0, 360, m, endpoint=False) - LIDAR_ANGLE_OFFSET_DEG
# 근데 measure_lidar_camera_offset.py로 실측한 결과(2026-08-14) 이 라이다는 한
# 바퀴에 n=500 포인트를 찍는 걸로 확인됐다. 정면(index≈80) 근처에선 이 옛 공식도
# 우연히 거의 맞지만(그 지점에서 캘리브레이션됐으므로), 정면에서 멀어질수록(=장애물이
# 옆에 있을수록) 각도 오차가 커진다 — 좌/우 장애물 판정, 라바콘 좌우 클러스터 판정 등
# "정면이 아닌 각도"를 보는 모든 라이다 판정에 영향을 줬을 가능성이 있다(README §2.30).
#
# local_costmap_proposal.md 선행작업 1순위: 프로덕션 코드를 실제 n 기반 공식으로
# 고치기 전에, n이 정말 "항상" 500인지(전원 재기동마다 안 바뀌는지, 스캔마다 안
# 흔들리는지)부터 확인해야 한다 — 이 도구가 그 확인을 자동화한다. 순수 구독(읽기)만
# 하고 아무것도 클릭/입력할 필요가 없어서 measure_lidar_camera_offset.py보다 훨씬
# 간단하다.
#
# ── 이 파일은 ROS2 빌드/노드 없이 그냥 실행합니다 ──
#   python3 measure_lidar_scan_count.py
# rclpy와 라이다 드라이버 노드(라이다만 떠 있으면 됨, track_drive 전체 launch
# 불필요)만 있으면 된다 — track_drive 패키지는 import하지 않는다.
#
# ── 사용법 ──
#   1) 라이다 드라이버만 띄운다.
#   2) python3 measure_lidar_scan_count.py [--duration 10]
#   3) 지정한 시간 동안 받은 모든 스캔의 len(ranges)를 집계해서, n이 일정한지/
#      스캔 주기(Hz)는 어떤지 + 프로덕션 코드에 적용할 수정 공식을 출력한다.
#=============================================
import argparse
import sys
import time
from collections import Counter


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--scan-topic', default='/scan')
    ap.add_argument('--duration', type=float, default=10.0, help='측정 시간(초)')
    args = ap.parse_args()

    try:
        import rclpy
        from rclpy.node import Node
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import LaserScan
    except ImportError:
        print('rclpy가 없습니다 — 이 도구는 실차(ROS2 환경)에서만 실행할 수 있습니다.')
        return 1

    counts = []
    timestamps = []

    class _Probe(Node):
        def __init__(self):
            super().__init__('measure_lidar_scan_count_probe')
            # xycar_lidar_node가 rclcpp::SensorDataQoS()(BEST_EFFORT)로 퍼블리시하므로
            # 구독 쪽도 동일 QoS로 맞춘다 — 안 맞추면 "New publisher discovered ...
            # incompatible QoS" 경고와 함께 메시지를 아예 못 받는다(measure_lidar_
            # camera_offset.py에서 이미 겪은 버그, README §2.30 참고).
            self.sub = self.create_subscription(
                LaserScan, args.scan_topic, self.cb, qos_profile_sensor_data)

        def cb(self, msg):
            counts.append(len(msg.ranges))
            timestamps.append(time.time())

    print(f'=== 라이다 스캔 포인트 개수(n) 실측 — {args.duration:.0f}초간 '
          f'{args.scan_topic} 구독 ===')
    rclpy.init(args=[])
    node = _Probe()
    t0 = time.time()
    try:
        while time.time() - t0 < args.duration:
            rclpy.spin_once(node, timeout_sec=0.2)
    finally:
        node.destroy_node()
        rclpy.shutdown()

    if not counts:
        print(f'{args.duration:.0f}초 안에 {args.scan_topic}에서 메시지를 못 받았습니다 — '
              '라이다 드라이버가 떠 있는지, 토픽 이름이 맞는지(--scan-topic) 확인하세요.')
        return 1

    hist = Counter(counts)
    n_unique = len(hist)
    mode_n, _mode_cnt = hist.most_common(1)[0]

    if len(timestamps) >= 2:
        dt = timestamps[-1] - timestamps[0]
        hz = (len(timestamps) - 1) / dt if dt > 0 else float('nan')
    else:
        hz = float('nan')

    print(f'\n총 스캔 수: {len(counts)}  (약 {hz:.1f}Hz)')
    print('n(=len(ranges)) 분포:')
    for n, cnt in sorted(hist.items()):
        print(f'  n={n:4d} : {cnt}회 ({100*cnt/len(counts):.1f}%)')

    print('\n=== 결과 ===')
    if n_unique == 1:
        print(f'OK — n이 이번 측정 내내 {mode_n}으로 일정합니다. 프로덕션 코드에서')
        print(f'  이 값을 상수로 가정해도 이번 측정 범위 안에서는 안전합니다')
        print(f'  (단, 재부팅/다른 라이다 개체에서도 같은지는 별도로 한 번 더 확인 권장).')
    else:
        print(f'주의 — n이 스캔마다 흔들립니다({n_unique}가지 값 관찰, 최빈값={mode_n}).')
        print(f'  프로덕션 코드는 고정 상수(500 등) 대신 매 스캔 len(msg.ranges)를')
        print(f'  그대로 읽어써야 안전합니다 — 아래 수정 방향의 "n"을 상수로 박지 말 것.')

    print('\ntrack_drive.py에 적용할 수정 방향 (perc_obstacle()/perc_lavacon_trigger() 둘 다):')
    print('  옛 공식(현재):')
    print('    m = min(len(ranges), 360)')
    print('    deg = linspace(0, 360, m, endpoint=False) - LIDAR_ANGLE_OFFSET_DEG')
    print('  → LIDAR_ANGLE_OFFSET_DEG(80.0)를 "각도"로 다뤄 매 인덱스에 똑같이 빼는데,')
    print('    실제로는 "정면일 때의 인덱스"로 실측된 값이라(README §6.2) n≠360이면')
    print('    정면에서 멀어질수록 오차가 커진다.')
    print('  새 공식(수정):')
    print('    n = len(ranges)   # 매 스캔 그대로, 상수로 고정하지 말 것')
    print('    deg = (index - LIDAR_ANGLE_OFFSET_DEG) * (360.0 / n)')
    print('  (measure_lidar_camera_offset.py의 _index_to_deg()와 동일 공식 — 그쪽은')
    print('   이미 고쳐져 있고 프로덕션 코드만 아직 안 고쳐진 상태, README §2.30 참고)')
    print('\n★주의★ perc_obstacle()은 이 각도 배열(self._obs_cos/_obs_sin)을 hasattr')
    print('  가드로 최초 1회만 계산해서 캐싱한다 — n이 스캔마다 안 바뀐다는 전제가')
    print('  깨지면(위에서 "주의"가 떴다면) 이 캐싱 로직도 같이 손봐야 한다.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
