#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#=============================================
# pp_tune_gridsearch.py — Pure Pursuit + 코너감속 파라미터 화이트박스 시뮬레이션
# 튜닝 도구 (2026-08-17)
#
# ── 왜 필요한가 ──
#   실차 회피 영상(2026-08-17 14:00)에서 "직진을 커브로 오판" 문제를 잡던 중,
#   현재 PP_* 튜닝값들이 SPEED_NORMAL=3.0(저속) 재튜닝 시점("대화 중 코너90도/S자
#   합성 시뮬레이션 그리드서치" — config.py PP_ALPHA/PP_DX_DEADZONE_PX 주석 참고)
#   결과라는 걸 확인했다. 그 시뮬레이션 자체는 코드로 저장되지 않고 대화 중에만
#   존재했던 것으로 보인다. SPEED_NORMAL이 이제 10.0으로 올랐고(§config.py
#   2026-08-17f) 나중에 더 올릴 계획이라, 이번엔 그 시뮬레이션 방법론을 코드로
#   남겨 재사용 가능하게 만들고 속도 10/20(모터단위)에서 다시 튜닝한다.
#
# ── [2026-08-17 2차] 코너 감속 로직까지 폐루프에 통합 ──
#   1차 버전은 시나리오 내내 speed_units를 상수로 고정해서 돌렸다 — 실제
#   track_drive.py._lane_drive()가 하는 "코너에서 조향 신호 기반으로 목표속도를
#   깎는" 로직(turn_for_speed 3제곱 감속식 + _corner_radius_speed_scale() 회전반경
#   기반 추가감속 + SPEED_ACCEL_STEP 가속램프 + CORNER_HOLD_DECAY 코너직후 감속유지)
#   이 전혀 반영 안 됐던 것. 조향(PP_*)과 감속(SPEED_*/CORNER_*)은 서로 커플링돼
#   있다 — 조향이 세지면 corner_signal이 커져 더 세게 감속하고, 감속하면 다음 틱
#   lookahead가 줄어 조향이 또 달라진다. 이번 버전은 simulate()가 _lane_drive()의
#   감속 공식을 그대로 재현해 매 틱 목표속도를 다시 계산하고, 그 값으로 물리
#   전진 + 다음 틱 lookahead 기준(speed_for_lookahead)까지 같이 갱신한다. 그리드서치
#   대상도 PP_*(조향) + SPEED_CORNER_MIN/CORNER_SIGN_EMA_ALPHA/LANE_LOOKAHEAD_REF/
#   SPEED_ACCEL_STEP/CORNER_HOLD_DECAY_LO·HI/CORNER_MIN_RADIUS_PX/
#   CORNER_MIN_SPEED_SCALE(감속)까지 22개 전부로 넓혔다.
#
# ── 방법론(화이트박스 합성 시뮬레이션 — 실측 아님) ──
#   controller/pure_pursuit.py의 PurePursuitController를 코드 수정 없이 그대로
#   가져다 쓴다(알고리즘 자체는 건드리지 않고 파라미터만 바꿔가며 평가). 직진/
#   90도커브/S자커브 3종 기준경로(월드 좌표, 미터)를 만들고, 매 제어틱(0.05s,
#   control_loop()과 동일 주기)마다
#     1) 차량의 현재 위치/헤딩 기준으로 기준경로를 차량 상대좌표로 변환
#     2) DL_PIXELS_PER_METER(200px/m)로 픽셀 변환 + 세그멘테이션 잡음(가우시안,
#        기본 표준편차 1.5px — config.py min_lookahead_px 주석의 실측 예
#        "dx=3px가 육안상 거의 직진"과 같은 자릿수)을 얹어 실제 ROI 픽셀
#        웨이포인트 배열(가까운점→먼점, PATH_N_WAYPOINTS=12개)을 합성
#     3) PurePursuitController.control()을 그대로 호출해 조향각(도)을 얻고
#     4) track_drive.py._lane_drive()/_corner_radius_speed_scale()과 동일한
#        공식으로 이번 틱 목표속도(모터단위)를 다시 계산하고(가속램프 포함)
#     5) 실측 WHEELBASE_M(0.335m, PP_WHEELBASE_PX와는 다른 값 — 아래 주의 참고)
#        기반 바이시클 모델로 그 속도만큼 차량을 실제로 전진시킨다
#   는 폐루프 시뮬레이션을 돌려, 기준경로 대비 횡편차(cross-track error)/조향
#   잔떨림/평균속도를 채점한다.
#
#   ★ 중요한 주의 — PP_WHEELBASE_PX ≠ WHEELBASE_M ★
#   PP_WHEELBASE_PX(현재 25.0)는 "곡률→조향각" 변환에 쓰이는 튜닝된 게인이지
#   차량의 실제 물리 축거가 아니다(config.py 주석: "튜닝값 — 조향을 더 ..."로
#   67.0(실측 기반 계산값)에서 낮춘 것). 즉 PurePursuitController 내부는
#   PP_WHEELBASE_PX를 쓰지만, 시뮬레이터가 차량을 실제로 움직일 때는 반드시
#   WHEELBASE_M(실측 축거)*바이시클 모델을 써야 한다 — 이 둘을 같은 값으로
#   섞으면 "튜닝된 게인이 물리적으로도 맞다"고 잘못 가정하게 되어 그리드서치가
#   자기 자신과 짜고 치는 꼴이 된다. _corner_radius_speed_scale()의 회전반경
#   계산은 실제 코드처럼 PP_WHEELBASE_PX(pp.wheelbase_px)를 그대로 쓴다 — 이건
#   물리 반경이 아니라 "그 함수가 보는" 반경이라 실제 코드와 똑같이 맞춰야 한다.
#
# ── 실차 미검증 명시 ──
#   이 도구가 찾아내는 "최적 파라미터"는 전부 화이트박스 합성 시뮬레이션
#   결과다. 실측이 아닌 값(DL_PIXELS_PER_METER=200px/m 자체가 설계값, 트랙
#   곡률 반경/노이즈 표준편차는 이 스크립트의 가정)에 의존하므로, 여기서 나온
#   조합은 반드시 실차에서 재검증할 것 — 기존 PP_* 값들과 동일한 원칙
#   (config.py 전역에 "실차 미검증" 표기가 붙는 것과 같은 이유).
#
# ── 속도 단위 ──
#   PurePursuitController.control(speed=...)의 speed는 "모터 단위"(drive()가
#   ±100으로 클립하는 값, m/s 아님) — lookahead 스케일링에만 쓰인다. 시뮬레이터가
#   차량을 실제로 전진시킬 때 쓰는 물리 속도(m/s)는 METERS_PER_SPEED_UNIT=0.1347
#   (config.py, speed=5~10 구간 실측 회귀)로 변환한다. speed=20은 이 회귀 구간
#   (5~10) 밖으로 외삽한 값이라 결과 해석 시 감안할 것 — README 6.5절 참고.
#   시나리오의 "speed=N"은 이제 SPEED_NORMAL(순항 목표속도) 역할이다 — 실제
#   순간속도는 코너에서 그보다 낮게 동적으로 깎인다(_lane_drive() 재현).
#
# ── VESC/IMU 미시뮬레이션 ──
#   실차의 _speed_for_lookahead()는 VESC 실측이 살아있으면 그걸 쓰지만, 이
#   시뮬레이터는 VESC/IMU를 흉내내지 않는다 — 실제 코드의 "VESC 죽었을 때" 폴백
#   경로(self._prev_speed 사용, imu_curvature_px=None)를 그대로 탄다. 즉 이
#   시뮬레이션은 "센서가 없거나 신뢰 못 할 때"의 동작을 검증하는 셈이고, VESC/IMU가
#   붙었을 때의 실제 거동은 여기서 다루지 않는다.
#
# ── 사용법 ──
#   numpy가 있는 환경에서:
#     python3 pp_tune_gridsearch.py --samples 3000 --speeds 10 20
#     python3 pp_tune_gridsearch.py --sensitivity --speeds 5 10 15 20 25
#=============================================
import argparse
import math
import sys
import os

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from controller.pure_pursuit import PurePursuitController  # noqa: E402
import config as cfg  # noqa: E402


# ─────────────────────────── 물리/스케일 상수 (config.py에서 그대로 가져옴) ───────────────────────────
PX_PER_M = cfg.DL_PIXELS_PER_METER          # 200.0
WHEELBASE_M = cfg.WHEELBASE_M               # 0.335 (실측 — PP_WHEELBASE_PX와 혼동 금지, 위 주석 참고)
MPS_PER_UNIT = cfg.METERS_PER_SPEED_UNIT    # 0.1347 (speed=5~10 실측 회귀)
ANGLE_MAX_DEG = cfg.ANGLE_MAX               # 80.0 — 안전 클램프, 그리드서치 대상 아님(고정)

ROI_W_PX = 585.0                            # README §2.35 실측: BEV 캔버스 실측폭
ROI_DEPTH_M = cfg.DL_BEV_FAR_LIMIT_M        # 0.7 — 원거리 크롭 한계(전방 가시거리 근사)
ROI_H_PX = ROI_DEPTH_M * PX_PER_M           # 140px — roi_h 근사(정확한 BEV 기하 대신 근사, 주석 참고)

N_WAYPOINTS = 12                            # lane_util.PATH_N_WAYPOINTS와 동일
# [알려진 한계] 1.5px는 "적당히 보수적인" 가정일 뿐 실측이 아니다 — 실제로 돌려보니
# 이 표준편차에서는 baseline(eps=0.001)조차 직진 시나리오에서 직진태그=100%가 나온다
# (즉 이 노이즈 수준만으로는 실차 영상에서 봤던 "직진을 커브로 오판" 실패모드가
# 재현되지 않는다 — 실제 세그멘테이션 잡음이 이보다 크거나, 순수 잡음이 아니라 지속적인
# 잔여 편향(bias)이 섞여있을 가능성). eps 자체를 이 시뮬레이션만으로 확정하지 말고,
# 이 한계를 감안해 실차 재검증할 것.
NOISE_STD_PX = 1.5                          # 세그멘테이션 프레임간 잡음 가정(표준편차, px)
DT = 0.05                                   # control_loop() 주기와 동일
MAX_SIM_T = 12.0                            # 시나리오별 최대 시뮬레이션 시간(초) — 발산 안전판
DIVERGE_CTE_M = 1.0                         # 이 이상 벗어나면 "발산"으로 보고 그 즉시 종료+페널티


# ─────────────────────────── 기준경로 생성 (월드 좌표, 미터, 표준 수학각) ───────────────────────────
def _straight(cursor, length, ds):
    x, y, th = cursor
    n = max(1, int(length / ds))
    pts = []
    for _ in range(n):
        x += ds * math.cos(th)
        y += ds * math.sin(th)
        pts.append((x, y))
    return pts, (x, y, th)


def _arc(cursor, radius, turn_deg, direction, ds):
    """direction: +1=좌회전(ccw), -1=우회전(cw). radius(m), turn_deg(회전각, 항상 양수로 줄 것)."""
    x, y, th = cursor
    total = math.radians(abs(turn_deg))
    dtheta = ds / radius
    n = max(1, int(total / dtheta))
    dtheta = total / n
    pts = []
    for _ in range(n):
        th += direction * dtheta
        x += ds * math.cos(th)
        y += ds * math.sin(th)
        pts.append((x, y))
    return pts, (x, y, th)


def build_paths(ds=0.02):
    """직진/90도커브/S자커브 3종 기준경로. 트랙 곡률 반경(1.2~1.3m)은 실측이 아니라
    VEHICLE_WIDTH_M(0.31)/LANE_WIDTH_M(0.4) 스케일에서 합리적으로 가정한 값 —
    실제 트랙 곡률과 다를 수 있으니 결과 해석 시 감안할 것."""
    paths = {}

    pts, _ = _straight((0.0, 0.0, math.pi / 2), 6.0, ds)
    paths['직진'] = pts

    cur = (0.0, 0.0, math.pi / 2)
    pts_a, cur = _straight(cur, 1.5, ds)
    pts_b, cur = _arc(cur, 1.2, 90.0, +1, ds)
    pts_c, cur = _straight(cur, 1.5, ds)
    paths['90도커브'] = pts_a + pts_b + pts_c

    cur = (0.0, 0.0, math.pi / 2)
    pts_a, cur = _straight(cur, 1.0, ds)
    pts_b, cur = _arc(cur, 1.3, 55.0, +1, ds)
    pts_c, cur = _arc(cur, 1.3, 55.0, -1, ds)
    pts_d, cur = _straight(cur, 1.0, ds)
    paths['S자커브'] = pts_a + pts_b + pts_c + pts_d

    return {k: np.array(v) for k, v in paths.items()}


# ─────────────────────────── 폐루프 시뮬레이션 ───────────────────────────
def _local_path_px(world_pts, vx, vy, vth, rng):
    """세계좌표 기준경로를 차량 상대좌표로 변환 → ROI 픽셀 웨이포인트(가까운점→먼점,
    N_WAYPOINTS개, 잡음 포함)로 합성. 실제 시스템의 _fit_and_sample_path()
    (균등 간격 재샘플링)와 동일한 발상."""
    dx = world_pts[:, 0] - vx
    dy = world_pts[:, 1] - vy
    forward = dx * math.cos(vth) + dy * math.sin(vth)
    lateral = dx * math.sin(vth) - dy * math.cos(vth)

    mask = (forward >= 0.0) & (forward <= ROI_DEPTH_M)
    if not np.any(mask):
        return None

    fwd_v = forward[mask]
    lat_v = lateral[mask]
    order = np.argsort(fwd_v)
    fwd_v, lat_v = fwd_v[order], lat_v[order]

    sample_fwd = np.linspace(0.0, min(fwd_v[-1], ROI_DEPTH_M), N_WAYPOINTS)
    sample_lat = np.interp(sample_fwd, fwd_v, lat_v)

    x_px = ROI_W_PX / 2.0 + sample_lat * PX_PER_M + rng.normal(0.0, NOISE_STD_PX, N_WAYPOINTS)
    y_px = ROI_H_PX - sample_fwd * PX_PER_M
    return list(zip(x_px.tolist(), y_px.tolist()))


def _cross_track_error(world_pts, vx, vy):
    d2 = (world_pts[:, 0] - vx) ** 2 + (world_pts[:, 1] - vy) ** 2
    return math.sqrt(float(np.min(d2)))


def _corner_radius_speed_scale(corner_signal_deg, wheelbase_px, corner_min_radius_px, corner_min_speed_scale):
    """track_drive.py._corner_radius_speed_scale()과 동일 공식. wheelbase_px는
    PP_WHEELBASE_PX(pp.wheelbase_px) — 실제 코드가 물리 축거가 아니라 이 게인으로
    반경을 역산하므로 그대로 맞춘다(위 모듈 주석 참고)."""
    curvature = math.tan(math.radians(corner_signal_deg)) / wheelbase_px
    if curvature == 0.0:
        return 1.0
    radius = abs(1.0 / curvature)
    if radius >= corner_min_radius_px:
        return 1.0
    return max(corner_min_speed_scale, radius / corner_min_radius_px)


def simulate(pp, world_pts, speed_norm, sp, rng):
    """speed_norm: 이 시나리오의 SPEED_NORMAL(모터단위, 순항 목표속도).
    sp: 감속 관련 파라미터 dict(SPEED_KEYS 참고) — track_drive.py._lane_drive()의
    코너 감속 로직을 그대로 재현한다."""
    vx, vy, vth = float(world_pts[0, 0]), float(world_pts[0, 1]), math.pi / 2
    pp.reset()

    prev_speed = 0.0        # track_drive.py __init__의 self._prev_speed=0.0과 동일 출발
    corner_signal = 0.0     # self._corner_signal
    corner_hold = 0.0       # self._corner_hold
    lookahead_ema = 0.0     # self.lane_lookahead(EMA, dl_lane far_ref 근사)

    ctes, steers, straight_flags, speeds = [], [], [], []
    path_end = world_pts[-1]
    n_ticks = int(MAX_SIM_T / DT)
    completed = False

    for _ in range(n_ticks):
        local_path = _local_path_px(world_pts, vx, vy, vth, rng)
        vehicle_xy = (ROI_W_PX / 2.0, ROI_H_PX)
        # _speed_for_lookahead(): VESC 미시뮬레이션이라 항상 self._prev_speed 폴백 경로.
        if local_path is None:
            steer_deg = pp.prev_steer_deg
        else:
            steer_deg = pp.control(local_path, vehicle_xy, speed=prev_speed, imu_curvature_px=None)
            far_offset_px = local_path[-1][0] - ROI_W_PX / 2.0
            lookahead_ema = 0.5 * lookahead_ema + 0.5 * far_offset_px

        # ── track_drive.py._lane_drive() 코너 감속 재현 ──
        corner_signal = (sp['corner_sign_ema_alpha'] * steer_deg
                          + (1.0 - sp['corner_sign_ema_alpha']) * corner_signal)
        turn_now = min(1.0, abs(corner_signal) / ANGLE_MAX_DEG)
        turn_preview = min(1.0, abs(lookahead_ema) / sp['lane_lookahead_ref'])
        is_straight = bool(pp.is_straight)
        turn_for_speed = 0.0 if is_straight else max(turn_now, turn_preview * 0.3)
        target_speed = max(sp['speed_corner_min'], speed_norm * (1.0 - 0.90 * turn_for_speed ** 3))
        corner_radius_scale = 1.0 if is_straight else _corner_radius_speed_scale(
            corner_signal, pp.wheelbase_px, sp['corner_min_radius_px'], sp['corner_min_speed_scale'])
        target_speed = max(sp['speed_corner_min'], target_speed * corner_radius_scale)

        speed_ratio = min(1.0, prev_speed / speed_norm) if speed_norm > 1e-6 else 0.0
        corner_decay = sp['corner_hold_decay_lo'] + (sp['corner_hold_decay_hi'] - sp['corner_hold_decay_lo']) * speed_ratio
        corner_hold = max(turn_now, corner_hold * corner_decay)
        accel_step = sp['speed_accel_step'] * max(0.25, 1.0 - corner_hold)
        if target_speed > prev_speed + accel_step:
            target_speed = prev_speed + accel_step
        prev_speed = target_speed

        # steer_deg는 "우측+"(control()의 alpha=atan2(dx,dy), dx>0=목표가 이미지 오른쪽일 때
        # 양수) 관례다. 월드 프레임 th는 표준 수학각(ccw+)이라, 오른쪽으로 꺾을수록(양의
        # steer) 헤딩은 시계방향=th가 "감소"해야 한다 — 부호를 안 뒤집으면 좌/우가 반대로
        # 시뮬레이션되어(첫 구현에서 실제로 발생: 커브가 좌회전인데 차가 우회전해 발산)
        # 그리드서치 전체가 무의미해진다.
        v_mps = MPS_PER_UNIT * prev_speed
        steer_rad = math.radians(steer_deg)
        vth -= (v_mps / WHEELBASE_M) * math.tan(steer_rad) * DT
        vx += v_mps * math.cos(vth) * DT
        vy += v_mps * math.sin(vth) * DT

        cte = _cross_track_error(world_pts, vx, vy)
        ctes.append(cte)
        steers.append(steer_deg)
        straight_flags.append(is_straight)
        speeds.append(prev_speed)

        if cte > DIVERGE_CTE_M:
            break
        if math.hypot(vx - path_end[0], vy - path_end[1]) < 0.15:
            completed = True
            break
    else:
        completed = math.hypot(vx - path_end[0], vy - path_end[1]) < 0.3

    ctes = np.array(ctes) if ctes else np.array([DIVERGE_CTE_M])
    steers = np.array(steers) if steers else np.array([0.0])
    speeds = np.array(speeds) if speeds else np.array([0.0])
    sign_changes = int(np.sum(np.diff(np.sign(steers)) != 0)) if len(steers) > 1 else 0
    osc_per_sec = sign_changes / max(len(steers) * DT, 1e-6)

    # 워밍업(≈1초, PP_STRAIGHT_CONFIRM_FRAMES 디바운스가 처음 채워지는 구간) 이후만 봐야
    # "계속 직진으로 안 잡히는" 문제를 제대로 측정한다 — 시작 직후 몇 틱은 아직 확정될
    # 시간이 없었을 뿐이라 여기서 빼지 않으면 모든 조합이 부당하게 벌점을 먹는다.
    warmup = int(1.0 / DT)
    tail = straight_flags[warmup:] if len(straight_flags) > warmup else straight_flags
    straight_frac = float(np.mean(tail)) if tail else 0.0

    return {
        'cte_rms': float(np.sqrt(np.mean(ctes ** 2))),
        'cte_max': float(np.max(ctes)),
        'steer_rms': float(np.sqrt(np.mean(steers ** 2))),
        'osc_per_sec': osc_per_sec,
        'straight_frac': straight_frac,
        'avg_speed': float(np.mean(speeds)),
        'elapsed_s': len(ctes) * DT,
        'completed': completed,
    }


# ─────────────────────────── 채점 ───────────────────────────
# 가중치는 이 스크립트만의 설계값(실측/이론적 근거 없음) — 직진은 "잔떨림 억제"를,
# 커브/S자는 "경로를 얼마나 잘 따라가는가(횡편차)"를 우선하되, elapsed_s에 작은
# 페널티를 줘서 "코너에서 무조건 기어가면 이긴다"는 퇴화해를 막는다(이 튜닝을
# 시작한 이유 자체가 "나중에 속도를 더 올리고 싶다"였으므로, 과도한 감속도
# 나쁜 해로 취급).
def score(results):
    s = results['직진']
    c = results['90도커브']
    w = results['S자커브']

    penalty = 0.0
    for r in (s, c, w):
        if not r['completed']:
            penalty += 1000.0

    # (1.0 - straight_frac)*10: 애초에 이번 튜닝을 시작한 이유("직진을 커브로 오판")를
    # 직접 벌점화한다 — steer_rms/osc/cte만으론 "라벨이 맞았는지"가 안 잡혀서 따로 추가.
    straight_score = (s['steer_rms'] * 3.0 + s['osc_per_sec'] * 2.0 + s['cte_rms'] * 50.0
                      + (1.0 - s['straight_frac']) * 10.0)
    curve_score = c['cte_rms'] * 80.0 + c['cte_max'] * 40.0 + c['elapsed_s'] * 0.5
    scurve_score = w['cte_rms'] * 80.0 + w['cte_max'] * 40.0 + w['elapsed_s'] * 0.5
    return straight_score + curve_score + scurve_score + penalty


# ─────────────────────────── 파라미터 공간 (baseline = 현재 config.py 값) ───────────────────────────
# PurePursuitController 생성자로 그대로 들어가는 키(조향) — evaluate()가 이 이름으로 필터링.
PP_CTOR_KEYS = (
    'lookahead_base_px', 'lookahead_speed_gain', 'lookahead_max_px', 'wheelbase_px',
    'alpha', 'min_lookahead_px', 'dx_deadzone_px', 'lookahead_curvature_gain',
    'lookahead_min_px', 'straight_curvature_eps', 'straight_confirm_frames',
    'straight_deadzone_px', 'straight_alpha', 'straight_bias_ema_alpha',
)
# track_drive.py._lane_drive()의 코너 감속 로직에 쓰이는 키(감속) — simulate()가 직접 읽음.
SPEED_KEYS = (
    'speed_corner_min', 'corner_sign_ema_alpha', 'lane_lookahead_ref',
    'speed_accel_step', 'corner_hold_decay_lo', 'corner_hold_decay_hi',
    'corner_min_radius_px', 'corner_min_speed_scale',
)

BASELINE = dict(
    lookahead_base_px=cfg.PP_LOOKAHEAD_BASE_PX,
    lookahead_speed_gain=cfg.PP_LOOKAHEAD_SPEED_GAIN,
    lookahead_max_px=cfg.PP_LOOKAHEAD_MAX_PX,
    wheelbase_px=cfg.PP_WHEELBASE_PX,
    alpha=cfg.PP_ALPHA,
    min_lookahead_px=cfg.PP_MIN_LOOKAHEAD_PX,
    dx_deadzone_px=cfg.PP_DX_DEADZONE_PX,
    lookahead_curvature_gain=cfg.PP_LOOKAHEAD_CURVATURE_GAIN,
    lookahead_min_px=cfg.PP_LOOKAHEAD_MIN_PX,
    straight_curvature_eps=cfg.PP_STRAIGHT_CURVATURE_EPS,
    straight_confirm_frames=cfg.PP_STRAIGHT_CONFIRM_FRAMES,
    straight_deadzone_px=cfg.PP_STRAIGHT_DEADZONE_PX,
    straight_alpha=cfg.PP_STRAIGHT_ALPHA,
    straight_bias_ema_alpha=cfg.PP_STRAIGHT_BIAS_EMA_ALPHA,
    speed_corner_min=cfg.SPEED_CORNER_MIN,
    corner_sign_ema_alpha=cfg.CORNER_SIGN_EMA_ALPHA,
    lane_lookahead_ref=cfg.LANE_LOOKAHEAD_REF,
    speed_accel_step=cfg.SPEED_ACCEL_STEP,
    corner_hold_decay_lo=cfg.CORNER_HOLD_DECAY_LO,
    corner_hold_decay_hi=cfg.CORNER_HOLD_DECAY_HI,
    corner_min_radius_px=cfg.CORNER_MIN_RADIUS_PX,
    corner_min_speed_scale=cfg.CORNER_MIN_SPEED_SCALE,
)

# (lo_factor, hi_factor) — baseline 대비 탐색 범위 배율. 정수/이산값(straight_confirm_frames)과
# 절대범위가 필요한 decay 두 값은 별도 처리(ABSOLUTE_RANGES).
RANGE_FACTORS = dict(
    lookahead_base_px=(0.7, 1.4),
    lookahead_speed_gain=(0.4, 1.8),
    lookahead_max_px=(0.8, 1.3),
    wheelbase_px=(0.5, 2.0),
    alpha=(0.4, 1.0 / max(cfg.PP_ALPHA, 1e-6) * 0.95 if cfg.PP_ALPHA < 0.95 else 1.0),
    min_lookahead_px=(0.7, 1.3),
    dx_deadzone_px=(0.3, 3.0),
    lookahead_curvature_gain=(0.3, 2.5),
    lookahead_min_px=(0.5, 1.8),
    straight_curvature_eps=(0.3, 3.0),
    straight_deadzone_px=(0.3, 2.0),
    straight_alpha=(0.4, 1.0 / max(cfg.PP_STRAIGHT_ALPHA, 1e-6) * 0.95 if cfg.PP_STRAIGHT_ALPHA < 0.95 else 1.0),
    straight_bias_ema_alpha=(0.4, 2.5),
    speed_corner_min=(0.4, 1.6),
    corner_sign_ema_alpha=(0.3, 3.0),
    lane_lookahead_ref=(0.5, 2.0),
    speed_accel_step=(0.3, 4.0),
    corner_min_radius_px=(0.4, 2.0),
    corner_min_speed_scale=(0.3, 2.5),
)
# 배율이 아니라 절대 범위가 필요한 파라미터 — (0,1) 안에 있어야 하고 baseline*factor로
# 다루면 방향에 따라 1을 넘거나 뒤집힐 수 있어서 따로 뺐다.
ABSOLUTE_RANGES = dict(
    corner_hold_decay_lo=(0.80, 0.95),
    corner_hold_decay_hi=(0.90, 0.99),
)
CONFIRM_FRAMES_CHOICES = [2, 3, 5, 7, 10]


def _candidate_range(name):
    if name in ABSOLUTE_RANGES:
        return ABSOLUTE_RANGES[name]
    lo_f, hi_f = RANGE_FACTORS[name]
    base = BASELINE[name]
    lo, hi = base * lo_f, base * hi_f
    if lo > hi:
        lo, hi = hi, lo
    return lo, hi


def sample_params(rng):
    p = {}
    for k, base in BASELINE.items():
        if k == 'straight_confirm_frames':
            p[k] = int(rng.choice(CONFIRM_FRAMES_CHOICES))
            continue
        lo, hi = _candidate_range(k)
        val = rng.uniform(lo, hi)
        if k in ('alpha', 'straight_alpha'):
            val = float(np.clip(val, 0.05, 1.0))
        p[k] = float(val)
    # corner_hold_decay_hi가 lo보다 낮게 뽑히면(별개 절대범위라 드물게 역전 가능)
    # 원래 설계 의도("고속일수록 느리게 회복")가 깨지므로 스왑해 방지.
    if p['corner_hold_decay_hi'] < p['corner_hold_decay_lo']:
        p['corner_hold_decay_lo'], p['corner_hold_decay_hi'] = p['corner_hold_decay_hi'], p['corner_hold_decay_lo']
    return p


def evaluate(params, speed_norm, paths, rng):
    pp_kwargs = {k: params[k] for k in PP_CTOR_KEYS}
    sp_kwargs = {k: params[k] for k in SPEED_KEYS}
    pp = PurePursuitController(angle_max_deg=ANGLE_MAX_DEG, **pp_kwargs)
    results = {name: simulate(pp, pts, speed_norm, sp_kwargs, rng) for name, pts in paths.items()}
    return score(results), results


def run_search(speed_norm, n_samples, seed, paths):
    rng = np.random.default_rng(seed)
    baseline_score, baseline_results = evaluate(BASELINE, speed_norm, paths, rng)

    best_score, best_params, best_results = baseline_score, dict(BASELINE), baseline_results
    for i in range(n_samples):
        params = sample_params(rng)
        s, results = evaluate(params, speed_norm, paths, rng)
        if s < best_score:
            best_score, best_params, best_results = s, params, results

    return {
        'speed_norm': speed_norm,
        'baseline_score': baseline_score,
        'baseline_results': baseline_results,
        'best_score': best_score,
        'best_params': best_params,
        'best_results': best_results,
    }


def sweep_param_vs_speed(param_name, speeds, grid_n, repeats, paths):
    """다른 파라미터는 전부 BASELINE에 고정하고 param_name 하나만 촘촘히 스윕해,
    speed별로 그 파라미터의 1차원 최적값을 찾는다(partial-dependence 근사 —
    파라미터 간 상호작용은 무시).

    [왜 조인트 랜덤서치 argmin 대신 이 방식인가] run_search()의 argmin은 22차원
    비볼록 공간에서 뽑은 단일 샘플이라, 같은 속도라도 시드만 바꾸면 전혀 다른
    조합이 "1등"으로 나올 만큼 분산이 크다(거의 동점인 국소 최적점이 많음).
    그런 고분산 추정치를 이어붙여 speed에 대한 추세선을 피팅하면 우연을 추세로
    착각하기 쉽다(실제로 speed=10/20 두 점만으로 봤을 때 PP_STRAIGHT_CURVATURE_EPS가
    방향이 뒤집혔던 게 그 예). 한 번에 파라미터 하나만 바꾸면 비교가 훨씬
    저분산이라 speed에 따른 진짜 추세를 보기에 더 적합하다."""
    if param_name == 'straight_confirm_frames':
        candidates = CONFIRM_FRAMES_CHOICES
    else:
        lo, hi = _candidate_range(param_name)
        candidates = np.linspace(lo, hi, grid_n)

    best_per_speed = []
    for sp in speeds:
        cand_scores = []
        for v in candidates:
            params = dict(BASELINE)
            params[param_name] = int(v) if param_name == 'straight_confirm_frames' else float(v)
            total = 0.0
            for rep in range(repeats):
                rng = np.random.default_rng(hash((param_name, sp, v, rep)) & 0xFFFFFFFF)
                s, _ = evaluate(params, sp, paths, rng)
                total += s
            cand_scores.append(total / repeats)
        best_idx = int(np.argmin(cand_scores))
        best_per_speed.append(float(candidates[best_idx]))
    return best_per_speed


def fit_linear(speeds, values):
    speeds_a, values_a = np.array(speeds, dtype=float), np.array(values, dtype=float)
    slope, intercept = np.polyfit(speeds_a, values_a, 1)
    pred = slope * speeds_a + intercept
    ss_res = float(np.sum((values_a - pred) ** 2))
    ss_tot = float(np.sum((values_a - np.mean(values_a)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 1.0
    return slope, intercept, r2


def run_sensitivity(speeds, grid_n, repeats, paths):
    print(f"[민감도 분석] 파라미터 1개씩만 스윕(다른 값은 BASELINE 고정), speed={speeds}, "
          f"grid_n={grid_n}, repeats={repeats}\n")
    fits = {}
    for name in BASELINE:
        best_vals = sweep_param_vs_speed(name, speeds, grid_n, repeats, paths)
        slope, intercept, r2 = fit_linear(speeds, best_vals)
        fits[name] = {'speeds': speeds, 'best_vals': best_vals,
                      'slope': slope, 'intercept': intercept, 'r2': r2}
        vals_str = ', '.join(f'{v:.4g}' for v in best_vals)
        print(f"  {name:26s} baseline={BASELINE[name]:<8.4g} @speeds→[{vals_str}]  "
              f"fit: y={slope:.5g}*speed+{intercept:.5g}  R²={r2:.2f}")
    return fits


def fmt_results(results):
    lines = []
    for name in ('직진', '90도커브', 'S자커브'):
        r = results[name]
        lines.append(
            f"    {name:6s}: cte_rms={r['cte_rms']*100:5.1f}cm cte_max={r['cte_max']*100:5.1f}cm "
            f"steer_rms={r['steer_rms']:5.2f}도 osc={r['osc_per_sec']:4.1f}/s "
            f"직진태그={r['straight_frac']*100:5.1f}% avg_speed={r['avg_speed']:5.2f} "
            f"소요={r['elapsed_s']:4.1f}s 완주={'Y' if r['completed'] else 'N'}"
        )
    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--samples', type=int, default=400, help='속도별 랜덤 샘플 수(기본 400)')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--speeds', type=float, nargs='+', default=[10.0, 20.0])
    ap.add_argument('--sensitivity', action='store_true',
                     help='조인트 랜덤서치 대신 파라미터별 1D 속도 민감도 분석(추세선 피팅용)')
    ap.add_argument('--grid-n', type=int, default=21, help='--sensitivity의 파라미터별 그리드 촘촘함')
    ap.add_argument('--repeats', type=int, default=3, help='--sensitivity의 후보값별 반복 평가 횟수(잡음 평균)')
    args = ap.parse_args()

    paths = build_paths()

    if args.sensitivity:
        return run_sensitivity(args.speeds, args.grid_n, args.repeats, paths)

    print(f"[설정] ROI_W={ROI_W_PX}px ROI_H≈{ROI_H_PX:.0f}px(DL_BEV_FAR_LIMIT_M={ROI_DEPTH_M}m 기준) "
          f"noise_std={NOISE_STD_PX}px WHEELBASE_M={WHEELBASE_M} MPS_PER_UNIT={MPS_PER_UNIT}")
    print(f"[baseline] SPEED_NORMAL=3.0 재튜닝 시점 config.py 현재값: {BASELINE}\n")

    all_out = {}
    for sp in args.speeds:
        print(f"===== speed={sp} (SPEED_NORMAL 역할, 모터unit, ≈{sp*MPS_PER_UNIT:.2f}m/s"
              f"{' — 5~10 회귀범위 밖 외삽' if sp > 10 else ''}) =====")
        out = run_search(sp, args.samples, args.seed, paths)
        all_out[sp] = out
        print(f"  baseline score={out['baseline_score']:.2f}")
        print(fmt_results(out['baseline_results']))
        print(f"  best({args.samples}샘플) score={out['best_score']:.2f}")
        print(fmt_results(out['best_results']))
        print("  best_params:")
        for k, v in out['best_params'].items():
            print(f"    {k} = {v:.4g}" if isinstance(v, float) else f"    {k} = {v}")
        print()

    return all_out


if __name__ == '__main__':
    main()
