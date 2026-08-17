#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#=============================================
# measure_speed_calibration.py — "찍히는 speed"(커맨드 단위 ctrl_speed / VESC 실측
# v_mps) vs 줄자+시간차로 잰 실제 지상속도(m/s) 비교 도구.
#
# ── 왜 필요한가 ──
# README §6.5의 METERS_PER_SPEED_UNIT(=0.1347)은 speed=5/10 단 2점 실측(각 2회 주행)만으로
# 낸 회귀값이라 "2점 외삽 한계"로 이미 명시돼 있고(x절편 speed≈1.4를 데드존으로 추정한 것도
# 실측이 아니라 추정치), VESC 실측(v_mps, README §7)도 지금까지 지상 실측(줄자)으로
# 교차검증된 적이 없다. 이 도구는 그 둘을 동시에, 여러 속도 지점에서 검증하기 위한 것.
#
# ── 원리 (사용자 계획 그대로) ──
# 트랙에 줄자로 등간격 지점을 표시해두고, 차량이 각 지점을 지날 때마다 Enter를 눌러
# 타임스탬프를 찍는다. 구간(mark[i-1]→mark[i])의 실제속도 = 구간거리 / 구간시간(뺄셈).
# 첫 구간(출발→mark1)은 초반가속이 섞여 있어 "정속" 가정이 깨지므로 참고용으로만 보여주고
# steady-state 평균/비교에서는 제외한다 — 이게 "초반가속을 고려한 뺄셈 연산"의 실제 구현.
# 그 구간 시간창 동안 로깅해둔 ctrl_speed/v_mps 평균과 실제속도를 나란히 비교해 오차%를 낸다.
#
# ── 측정 오차에 대한 주의 ──
# 사람이 Enter를 누르는 반응시간 오차(대략 0.1~0.2초)가 있다. 구간 시간이 1초 미만이면
# 이 오차가 전체의 10%+ 를 차지할 수 있다 — 기본 간격을 35cm로 좁게 잡은 만큼(예상 dt가
# 1초를 한참 밑돎) **개별 구간 숫자 하나하나는 잡음이 크다고 보고, 아래 두 가지 보정을
# 반드시 같이 볼 것**:
#   ① 가속구간 자동감지 — "첫 구간만 가속"이라고 가정하지 않는다. 연속 2개 구간의
#      실제속도가 서로 12% 이내로 비슷해지는 지점부터 steady로 판단한다(35cm 간격이면
#      가속이 여러 구간에 걸쳐 있을 수 있어서).
#   ② 회귀 기반 steady 속도 — steady로 판단된 구간의 각 Enter 시점(t, 누적거리)들을
#      최소자승 회귀(기울기=속도)로 한 번에 묶어서 낸 값. 개별 Enter 하나의 반응시간
#      잡음이 상쇄되므로, 구간별 표보다 이 값을 최종 수치로 더 신뢰할 것.
#
# ── 두 가지 모드 ──
# 1) 패시브 모드(기본, --drive-speed 생략): 이미 떠 있는 다른 노드(track_drive.py 등)가
#    xycar_motor를 발행 중일 때 그 값만 구독해서 관찰. 실제 주행(차선추종 등)의 속도
#    프로파일을 검증할 때 사용.
# 2) 능동 모드(--drive-speed X): 이 도구가 직접 angle=0, speed=X를 xycar_motor로 발행해
#    "직진 정속 테스트"를 만든다(가속 램프는 config.py SPEED_ACCEL_STEP과 동일하게 재현).
#
#    ★★★ 안전 — 능동 모드는 차선/장애물 인식이 전혀 없는 완전 개루프 주행이다 ★★★
#      - track_drive.py(또는 xycar_motor를 발행하는 다른 노드)는 반드시 꺼둘 것
#        (같은 토픽에 두 발행자가 붙으면 명령이 뒤섞인다)
#      - 사람이 없는 직선 트랙에서만, 즉시 멈출 수 있게 준비할 것(Ctrl+C 또는 리모컨
#        킬스위치) — Ctrl+C를 누르면 즉시 speed=0을 발행하고 종료한다
#      - --max-duration(기본 10초)이 지나면 마크 입력과 무관하게 자동으로 speed=0 발행
#
# ── 사용법(능동 모드 예시) ──
#   python3 measure_speed_calibration.py --drive-speed 15
#   (--interval-m 기본값 0.35 그대로 사용 — 줄자로 35cm 간격 표시를 미리 해둔다)
#   1) 차량을 정지선 뒤에 두고, 그 지점부터 줄자로 35cm 간격 표시를 미리 해둔다.
#   2) 스크립트 실행 → "0m 지점(출발) — 준비되면 Enter" 프롬프트에서 Enter →
#      그 즉시 차량이 speed=15 목표로 출발(가속램프 적용)하고 mark0(0m)가 기록된다.
#   3) 차량이 각 표시 지점(0.35m, 0.7m, 1.05m, ...)을 지나는 순간마다 Enter.
#   4) 다 지났으면 'q' + Enter로 종료 → 자동 speed=0 발행 + 구간별 결과표 출력 +
#      CSV 저장. 터미널에 출력된 표를 그대로 복사해 Claude에게 붙여넣으면 취합/판단.
#
# ── 사용법(패시브 모드) ──
#   python3 measure_speed_calibration.py
#   (track_drive.py를 정상 실행해둔 상태에서, 직선 구간을 지날 때 마크 Enter만 누르면 됨 —
#    첫 Enter가 0m 기준점이라, 반드시 차량이 표시해둔 0m 지점을 지나는 순간 눌러야 한다)
#
# ── config.py와 동기화 유지 ──
# 아래 상수들은 config.py에서 그대로 복사해 왔다(measure_lidar_camera_offset.py와 같은
# 관례 — 이 도구는 track_drive 패키지를 import하지 않고 독립 실행됨). config.py에서 값이
# 바뀌면 여기도 같이 갱신할 것.
#=============================================
import argparse
import csv
import datetime
import os
import statistics
import threading
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import Float32, Float32MultiArray

# ── config.py와 반드시 동기화 유지 ──
METERS_PER_SPEED_UNIT = 0.1347     # README §6.5, speed 5~10 2점 회귀(현재값, 비교 기준선)
VESC_SPEED_TO_ERPM_GAIN = 4614.0   # vesc.yaml speed_to_erpm_gain 실측값
SPEED_ACCEL_STEP = 0.4             # 가속 램프(주기당 최대 증가량) — control_loop과 동일 주기 가정
CONTROL_PERIOD_SEC = 0.05          # track_drive.py control_loop 주기(20Hz)와 동일


class SpeedCalibNode(Node):
    def __init__(self, motor_topic, vesc_topic, drive_speed, max_duration):
        super().__init__('measure_speed_calibration')
        self.drive_speed = drive_speed
        self.max_duration = max_duration

        self._lock = threading.Lock()
        self.ctrl_samples = []   # [(t, ctrl_speed), ...] — xycar_motor에서 관측
        self.vesc_samples = []   # [(t, v_mps), ...]

        self.create_subscription(Float32MultiArray, motor_topic, self._cb_motor, 10)
        self.create_subscription(Float32, vesc_topic, self._cb_vesc, qos_profile_sensor_data)

        self._motor_pub = None
        self._drive_t0 = None
        self._cur_speed = 0.0
        self._force_stopped = False
        if drive_speed is not None:
            self._motor_pub = self.create_publisher(Float32MultiArray, motor_topic, 10)
            self.create_timer(CONTROL_PERIOD_SEC, self._drive_step)

    def _cb_motor(self, msg):
        if len(msg.data) < 2:
            return
        with self._lock:
            self.ctrl_samples.append((time.time(), float(msg.data[1])))

    def _cb_vesc(self, msg):
        with self._lock:
            self.vesc_samples.append((time.time(), float(msg.data) / VESC_SPEED_TO_ERPM_GAIN))

    def start_drive(self, t0):
        self._drive_t0 = t0

    def _drive_step(self):
        if self._drive_t0 is None or self._motor_pub is None:
            return
        elapsed = time.time() - self._drive_t0
        if elapsed >= self.max_duration:
            if not self._force_stopped:
                self._force_stopped = True
                self.get_logger().warn(
                    f'[안전정지] --max-duration({self.max_duration:.1f}s) 경과 — speed=0 강제 발행')
            self._cur_speed = 0.0
        else:
            target = self.drive_speed
            step = SPEED_ACCEL_STEP
            if self._cur_speed < target:
                self._cur_speed = min(target, self._cur_speed + step)
            elif self._cur_speed > target:
                self._cur_speed = max(target, self._cur_speed - step)
        self._publish(0.0, self._cur_speed)

    def _publish(self, angle, speed):
        # track_drive.py의 drive()와 동일하게 주기당 7번 반복 발행 — 1번만 보내면
        # ros1_bridge/ESC 쪽 패킷 유실·워치독 타임아웃으로 명령 사이사이 모터가 꺼져
        # "찍히긴 하는데 못 나가는" 증상이 나는 것으로 실차에서 확인됨(2026-08-17).
        msg = Float32MultiArray()
        msg.data = [float(angle), float(max(-100.0, min(100.0, speed)))]
        for _ in range(7):
            self._motor_pub.publish(msg)

    def stop_motor(self):
        if self._motor_pub is not None:
            self._publish(0.0, 0.0)

    def samples_snapshot(self):
        with self._lock:
            return list(self.ctrl_samples), list(self.vesc_samples)


def _input_worker(node, interval_m, markers, stop_event):
    """마크 입력 전담 스레드. 첫 Enter가 0m 기준점(+능동모드면 출발 트리거)이고,
    이후 매 Enter가 다음 마크. 숫자를 입력하면 그 마크의 절대거리(m)를 균일간격 대신
    직접 지정(간격이 고르지 않은 트랙 사정 대응용), 'q'는 종료."""
    if node.drive_speed is not None:
        print(f'\n[능동모드] 0m 지점(출발선)에 차량을 두고 준비되면 Enter — '
              f'즉시 speed={node.drive_speed:.1f} 목표로 출발합니다.')
    else:
        print('\n[패시브모드] 차량이 0m 기준점을 지나는 순간 Enter.')
    try:
        input()
    except EOFError:
        stop_event.set()
        return
    t0 = time.time()
    markers.append({'idx': 0, 't': t0, 'dist_m': 0.0})
    if node.drive_speed is not None:
        node.start_drive(t0)

    idx = 1
    while not stop_event.is_set():
        try:
            line = input(f'  mark{idx} ({idx * interval_m:.1f}m 예상, 숫자=실측거리 직접입력, '
                          f'q=종료): ').strip()
        except EOFError:
            break
        if line.lower() == 'q':
            break
        dist = idx * interval_m
        if line:
            try:
                dist = float(line)
            except ValueError:
                print('    (숫자로 해석 안 됨 — 균일 간격으로 처리)')
        markers.append({'idx': idx, 't': time.time(), 'dist_m': dist})
        idx += 1
    stop_event.set()


def _mean(vals):
    return statistics.fmean(vals) if vals else None


def _detect_steady_start(rows, rel_tol=0.12, consec=2):
    """rows(순서대로)에서 steady-state가 시작되는 인덱스(0-based)를 찾는다.
    "첫 구간만 가속"이라고 하드코딩하지 않는 이유: --interval-m을 35cm처럼 좁게 잡으면
    가속이 한 구간 안에서 안 끝나고 여러 구간에 걸칠 수 있다. 연속 consec개 구간의
    실제속도가 서로 rel_tol(기본 12%) 이내로 수렴하는 첫 지점을 steady 시작으로 본다 —
    휴리스틱이라 완벽하진 않으니, 출력되는 구간별 표를 보고 이상하면 직접 판단할 것."""
    if len(rows) < consec:
        return len(rows)
    for i in range(len(rows) - consec + 1):
        vals = [r['real_mps'] for r in rows[i:i + consec]]
        vmean = statistics.fmean(vals)
        if vmean > 0 and (max(vals) - min(vals)) <= rel_tol * vmean:
            return i
    return len(rows)


def _regression_speed(marks):
    """steady 구간에 속한 마크들(t, dist_m)의 최소자승 회귀 기울기 = 속도(m/s).
    개별 Enter 하나의 반응시간 잡음이 여러 점에 걸쳐 상쇄되므로, 구간(2점)만으로 낸
    real_mps보다 잡음에 강건하다 — 35cm처럼 좁은 간격에서 특히 유용."""
    if len(marks) < 3:
        return None
    ts = [m['t'] for m in marks]
    ds = [m['dist_m'] for m in marks]
    tbar, dbar = _mean(ts), _mean(ds)
    den = sum((t - tbar) ** 2 for t in ts)
    if den <= 0:
        return None
    return sum((t - tbar) * (d - dbar) for t, d in zip(ts, ds)) / den


def _analyze(markers, ctrl_samples, vesc_samples, out_dir, drive_speed):
    if len(markers) < 2:
        print('\n마크가 2개 미만이라 분석할 구간이 없습니다.')
        return

    rows = []
    for i in range(1, len(markers)):
        m0, m1 = markers[i - 1], markers[i]
        dt = m1['t'] - m0['t']
        dd = m1['dist_m'] - m0['dist_m']
        if dt <= 0 or dd <= 0:
            continue
        real_mps = dd / dt
        ctrl_win = [s for (t, s) in ctrl_samples if m0['t'] <= t <= m1['t']]
        vesc_win = [s for (t, s) in vesc_samples if m0['t'] <= t <= m1['t']]
        ctrl_avg = _mean(ctrl_win)
        vesc_avg = _mean(vesc_win)
        pred_mps = ctrl_avg * METERS_PER_SPEED_UNIT if ctrl_avg is not None else None
        implied_unit = (real_mps / ctrl_avg) if ctrl_avg else None
        rows.append(dict(seg=i, dt=dt, dd=dd, real_mps=real_mps,
                          ctrl_avg=ctrl_avg, vesc_avg=vesc_avg, pred_mps=pred_mps,
                          implied_unit=implied_unit, n_ctrl=len(ctrl_win), n_vesc=len(vesc_win)))

    steady_start_idx = _detect_steady_start(rows)
    for j, r in enumerate(rows):
        r['accel_phase'] = j < steady_start_idx
    if steady_start_idx >= len(rows):
        print('\n⚠ steady 구간을 자동감지하지 못했습니다(끝까지 가속/변속 중으로 보임) — '
              '마크를 더 찍거나(더 먼 거리) 아래 표를 보고 직접 steady 구간을 판단하세요.')
    else:
        print(f'\nsteady 시작 자동감지: segment {steady_start_idx + 1}부터 '
              f'(연속 2구간 실제속도가 12% 이내로 수렴하는 지점)')

    print('\n=== 구간별 결과 ===')
    header = (f'{"seg":>4} {"구간(m)":>8} {"dt(s)":>7} {"실제(m/s)":>10} '
              f'{"ctrl_avg":>9} {"pred(cfg)":>10} {"vesc_avg":>9} {"단위당(m/s)":>11}')
    print(header)
    for r in rows:
        tag = '가속구간*' if r['accel_phase'] else ''
        print(f'{r["seg"]:>4} {r["dd"]:>8.2f} {r["dt"]:>7.3f} {r["real_mps"]:>10.3f} '
              f'{r["ctrl_avg"] if r["ctrl_avg"] is not None else float("nan"):>9.2f} '
              f'{r["pred_mps"] if r["pred_mps"] is not None else float("nan"):>10.3f} '
              f'{r["vesc_avg"] if r["vesc_avg"] is not None else float("nan"):>9.3f} '
              f'{r["implied_unit"] if r["implied_unit"] is not None else float("nan"):>11.4f} {tag}')
        if r['dt'] < 1.0:
            print(f'      ⚠ 구간 시간 {r["dt"]:.2f}s < 1초 — 사람 반응시간 오차(~0.1~0.2s)가 '
                  f'{r["dt"]:.2f}s의 10%+ 를 차지할 수 있습니다. 간격을 넓히거나 반복측정 권장.')
        if r['n_ctrl'] == 0:
            print('      ⚠ 이 구간엔 ctrl_speed 샘플이 없습니다(토픽 미수신?) — ctrl_avg 신뢰 불가.')
        if r['n_vesc'] == 0:
            print('      ⚠ 이 구간엔 VESC 샘플이 없습니다(vesc_speed_bridge 미실행/브리지 끊김?).')

    steady = [r for r in rows if not r['accel_phase']]
    print('\n=== steady-state 요약(자동감지된 가속구간 제외) ===')
    if steady:
        real_vals = [r['real_mps'] for r in steady]
        ctrl_vals = [r['ctrl_avg'] for r in steady if r['ctrl_avg'] is not None]
        vesc_vals = [(r['real_mps'], r['vesc_avg']) for r in steady if r['vesc_avg'] is not None]
        unit_vals = [r['implied_unit'] for r in steady if r['implied_unit'] is not None]
        print(f'steady 구간 수: {len(steady)}')
        print(f'실제속도 평균={_mean(real_vals):.3f} m/s '
              f'(표준편차={statistics.pstdev(real_vals):.3f}, 편차가 크면 아직도 가속/감속 중이란 뜻)')
        if ctrl_vals:
            print(f'ctrl_speed 평균={_mean(ctrl_vals):.2f}')
        if unit_vals:
            implied = _mean(unit_vals)
            cfg_err_pct = abs(implied - METERS_PER_SPEED_UNIT) / METERS_PER_SPEED_UNIT * 100
            print(f'실측 기반 METERS_PER_SPEED_UNIT ≈ {implied:.4f} '
                  f'(현재 config.py 값 {METERS_PER_SPEED_UNIT} 대비 오차 {cfg_err_pct:.1f}%)')
        start_pos = rows[steady_start_idx]['seg'] - 1 if steady_start_idx < len(rows) else None
        if start_pos is not None:
            reg_speed = _regression_speed(markers[start_pos:])
            if reg_speed is not None:
                print(f'회귀 기반 steady 실제속도(Enter 반응시간 잡음에 강건) = {reg_speed:.3f} m/s '
                      f'(마크 {len(markers) - start_pos}개 사용) ← 35cm처럼 좁은 간격이면 이 값을 '
                      f'구간별 표보다 우선 신뢰할 것')
                if ctrl_vals:
                    implied_reg = reg_speed / _mean(ctrl_vals)
                    print(f'  → 회귀 기반 METERS_PER_SPEED_UNIT ≈ {implied_reg:.4f}')
        if vesc_vals:
            errs = [abs(v - r) / r * 100 for r, v in vesc_vals if r > 0]
            if errs:
                print(f'VESC(v_mps) vs 실제 평균 오차 = {_mean(errs):.1f}% (n={len(errs)})')
    else:
        print('전 구간이 가속구간으로 판단돼 steady 구간이 없습니다 — 마크를 더 찍어보세요.')

    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    tag = f'drive{drive_speed:g}' if drive_speed is not None else 'passive'
    csv_path = os.path.join(out_dir, f'speed_calib_{tag}_{ts}.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['seg', 'accel_phase', 'segment_m', 'dt_s', 'real_mps', 'ctrl_avg',
                     'pred_mps_cfg', 'vesc_avg_mps', 'implied_meters_per_unit', 'n_ctrl', 'n_vesc'])
        for r in rows:
            w.writerow([r['seg'], r['accel_phase'], r['dd'], r['dt'], r['real_mps'],
                        r['ctrl_avg'], r['pred_mps'], r['vesc_avg'], r['implied_unit'],
                        r['n_ctrl'], r['n_vesc']])
    print(f'\nCSV 저장: {csv_path}')
    print('(위 "구간별 결과"/"steady-state 요약" 콘솔 출력을 그대로 복사해 Claude에게 붙여넣으면 됩니다.)')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--interval-m', type=float, default=0.35,
                     help='마크 간 기본 간격(m), 균일 간격 가정 시 사용(기본 0.35=35cm — '
                          '짧을수록 개별 구간은 잡음이 커지니 steady 구간은 회귀값을 볼 것)')
    ap.add_argument('--drive-speed', type=float, default=None,
                     help='능동모드: 이 값으로 angle=0 직진 정속 명령을 직접 발행. '
                          '생략하면 패시브(관찰만) 모드.')
    ap.add_argument('--max-duration', type=float, default=10.0,
                     help='능동모드 안전 컷오프(초) — 경과 시 강제로 speed=0 (기본 10.0)')
    ap.add_argument('--motor-topic', default='xycar_motor')
    ap.add_argument('--vesc-topic', default='/vesc_speed_erpm')
    ap.add_argument('--out-dir', default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                        'speed_calib_logs'))
    args = ap.parse_args()

    if args.drive_speed is not None:
        print('=' * 60)
        print('★★★ 능동모드 — 완전 개루프 직진 주행 ★★★')
        print('- track_drive.py 등 xycar_motor를 발행하는 다른 노드가 꺼져 있는지 확인하세요.')
        print('- 사람이 없는 직선 트랙에서만 실행하고, Ctrl+C로 언제든 즉시 정지시킬 수 있습니다.')
        print(f'- {args.max_duration:.1f}초가 지나면 자동으로 speed=0이 발행됩니다.')
        print('=' * 60)

    rclpy.init(args=[])
    node = SpeedCalibNode(args.motor_topic, args.vesc_topic, args.drive_speed, args.max_duration)

    markers = []
    stop_event = threading.Event()
    th = threading.Thread(target=_input_worker, args=(node, args.interval_m, markers, stop_event),
                           daemon=True)
    th.start()

    try:
        while not stop_event.is_set():
            rclpy.spin_once(node, timeout_sec=0.05)
    except KeyboardInterrupt:
        print('\n[Ctrl+C] 즉시 정지합니다.')
    finally:
        node.stop_motor()
        for _ in range(3):   # 정지 명령이 확실히 나가도록 몇 차례 더 publish
            rclpy.spin_once(node, timeout_sec=0.05)
        ctrl_samples, vesc_samples = node.samples_snapshot()
        node.destroy_node()
        rclpy.shutdown()

    _analyze(markers, ctrl_samples, vesc_samples, args.out_dir, args.drive_speed)


if __name__ == '__main__':
    main()
