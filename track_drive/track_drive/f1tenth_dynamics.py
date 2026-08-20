#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#=============================================
# f1tenth_dynamics.py — F1TENTH Gym의 차량 동역학 함수 이식 (2026-08-18)
#
# ── 왜 필요한가 ──
#   pp_tune_gridsearch.py의 기존 물리 스텝(simulate() 안의 vx/vy/vth 업데이트)은 순수
#   기구학(kinematic) 자전거모델이다 — 타이어가 항상 완벽하게 그립한다고 가정하므로
#   슬립을 원리상 표현할 수 없다. 실차에서 보고된 "고속 코너에서 술취한 듯 도는" 진동이
#   타이어 슬립/조향 액추에이터 지연에서 온다면, 기구학 모델로는 그 실패모드 자체를
#   시뮬레이션에서 재현할 수 없어 아무리 그리드서치를 돌려도 못 잡는다(대화 중 원인
#   진단 참고). 이 파일은 F1TENTH Gym(github.com/f1tenth/f1tenth_gym)의
#   gym/f110_gym/envs/dynamic_models.py에 있는 Single Track Dynamic 모델(타이어
#   코너링 강성 기반 슬립 반영)을 그대로 가져와, pp_tune_gridsearch.py가 선택적으로
#   (--physics st) 쓸 수 있게 한다.
#
# ── 출처/라이선스 ──
#   원본: dynamic_models.py, Copyright 2020 Technical University of Munich,
#   Professorship of Cyber-Physical Systems, Matthew O'Kelly, Aman Sinha, Hongrui Zheng.
#   BSD 3-Clause 스타일 라이선스(재배포 시 저작권고지 유지 조건, 하단 원문 보존) —
#   MIT가 아님에 주의(같은 저장소의 f110_env.py는 MIT지만 이 파일 출처는 다른 라이선스).
#   commonroad-vehicle-models(TUM)의 Single Track 동역학 모델을 따른 것이라고 원본
#   docstring에 명시돼 있음(원 저자: Hongrui Zheng).
#
#   아래 accl_constraints/steering_constraint/vehicle_dynamics_ks/vehicle_dynamics_st/pid
#   5개 함수는 원본과 수학적으로 동일하다(변수명/로직 그대로, 검증용 벡터도 원본
#   DynamicsTest와 대조 가능) — @njit(numba) 데코레이터만 "numba 없으면 그냥 순수
#   파이썬으로 동작"하도록 optional로 바꿨다(이 저장소 개발환경엔 numba/numpy조차
#   기본 설치돼 있지 않음 — CLAUDE.md "이 환경엔 ROS2 툴체인이 없음" 참고, 실행은
#   numpy가 있는 별도 환경에서 해야 한다).
#
# ── 물리 파라미터 기본값(F1TENTH_DEFAULT_PARAMS) 출처 ──
#   F1TENTH Gym의 f110_env.py가 실제 1/10 스케일 차량 기본값으로 쓰는 딕셔너리를
#   그대로 가져왔다(dynamic_models.py 자체의 DynamicsTest.setUp() 값은 쓰지 않았음 —
#   그건 commonroad 원 논문의 "Vehicle 2"(BMW 320i) 파라미터를 피트/파운드에서
#   환산한 실차 스케일 값이라 1/10 RC카에 안 맞는다, m≈1092kg로 계산됨).
#   F1TENTH 기본값의 lf+lr = 0.15875+0.17145 = 0.3302m — 우리 실측 WHEELBASE_M(0.335,
#   §6.7)과 1.5%밖에 안 틀려서 같은 체급(1/10 스케일 RC 섀시)이라는 정황상 신뢰도가
#   높다. 반면 mu/C_Sf/C_Sr/h/m/I는 F1TENTH 자체 차량 값이라 xycar 실측이 아니다 —
#   ★ 실차 미검증, F1TENTH 대체값 ★. 나중에 xycar 질량/타이어를 실측하면 교체할 것.
#=============================================
import math

try:
    from numba import njit
except ImportError:
    def njit(*args, **kwargs):
        """numba 미설치 환경용 no-op 대체 — @njit(cache=True) 형태(인자 있음)와
        @njit 형태(인자 없음) 둘 다 지원."""
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]

        def _wrap(fn):
            return fn
        return _wrap


# F1TENTH Gym f110_env.py 기본 차량 파라미터(1/10 스케일) — 위 모듈 주석 "물리 파라미터
# 기본값 출처" 참고. lf/lr만 우리 실측과 정합적으로 검증됨, 나머지는 F1TENTH 대체값.
F1TENTH_DEFAULT_PARAMS = {
    'mu': 1.0489,      # 노면-타이어 마찰계수 — xycar 미실측, F1TENTH 대체값
    'C_Sf': 4.718,     # 앞바퀴 코너링 강성 — xycar 미실측, F1TENTH 대체값
    'C_Sr': 5.4562,    # 뒷바퀴 코너링 강성 — xycar 미실측, F1TENTH 대체값
    'lf': 0.15875,     # 무게중심→앞축 거리(m)
    'lr': 0.17145,     # 무게중심→뒷축 거리(m) — lf+lr=0.3302m, WHEELBASE_M(0.335)과 1.5% 오차
    'h': 0.074,        # 무게중심 높이(m) — xycar 미실측, F1TENTH 대체값
    'm': 3.74,         # 질량(kg) — xycar 미실측, F1TENTH 대체값
    'I': 0.04712,      # 요(yaw) 관성모멘트(kg*m^2) — xycar 미실측, F1TENTH 대체값
    # [2026-08-18 실차 확인 반영] F1TENTH 기본값(±0.4189rad≈24도, 그쪽 서보 스펙)이 아니라
    #   우리 실측으로 교체 — 실차에서 ANGLE_MAX=80도로 정상 주행 중이라고 확인됨(즉 80도가
    #   "명령 스케일값"이 아니라 실제로 서보가 그만큼 꺾인다는 뜻, 위 [알려진 한계] 해소).
    #   덧붙여 조향각을 소프트웨어에서 50도로 제한해도 우리 트랙 전 구간(90도커브+지그재그
    #   포함) 완주에 문제없었다는 실차 확인도 있음 — 즉 물리적 최대각은 >=80도지만
    #   "이 트랙을 도는 데 실제로 필요한" 조향각은 50도면 충분하다는 뜻. s_min/s_max는
    #   "물리적으로 꺾일 수 있는 한계"라 여기선 실측 확인된 80도를 그대로 쓴다 — 50도는
    #   config.py ANGLE_MAX를 낮추는 실험(고조향각 구간에서의 슬립/진동 감소 가설, 대화
    #   참고)의 후보값이지 이 물리모델의 하드 리밋으로 쓸 값은 아니다.
    's_min': -math.radians(80.0),
    's_max': math.radians(80.0),
    'sv_min': -3.2,    # 최소 조향각속도(rad/s)
    'sv_max': 3.2,     # 최대 조향각속도(rad/s) — 이게 "조향 액추에이터 반응지연"을 만드는 값
    'v_switch': 7.319, # 이 속도 이상에서는 가속도가 v_switch/v에 비례해 줄어듦(휠스핀 한계)
    'a_max': 9.51,     # 최대 가속도(m/s^2)
    'v_min': -5.0,     # 최소 속도(후진, m/s) — 우리는 전진만 명령하므로 사실상 안 쓰임
    'v_max': 20.0,     # 최대 속도(m/s)
}


@njit(cache=True)
def accl_constraints(vel, accl, v_switch, a_max, v_min, v_max):
    """가속도 제약 — 원본 dynamic_models.py와 동일."""
    if vel > v_switch:
        pos_limit = a_max * v_switch / vel
    else:
        pos_limit = a_max

    if (vel <= v_min and accl <= 0) or (vel >= v_max and accl >= 0):
        accl = 0.
    elif accl <= -a_max:
        accl = -a_max
    elif accl >= pos_limit:
        accl = pos_limit

    return accl


@njit(cache=True)
def steering_constraint(steering_angle, steering_velocity, s_min, s_max, sv_min, sv_max):
    """조향각속도 제약 — 원본 dynamic_models.py와 동일."""
    if (steering_angle <= s_min and steering_velocity <= 0) or (steering_angle >= s_max and steering_velocity >= 0):
        steering_velocity = 0.
    elif steering_velocity <= sv_min:
        steering_velocity = sv_min
    elif steering_velocity >= sv_max:
        steering_velocity = sv_max

    return steering_velocity


@njit(cache=True)
def vehicle_dynamics_ks(x, u_init, mu, C_Sf, C_Sr, lf, lr, h, m, I,
                         s_min, s_max, sv_min, sv_max, v_switch, a_max, v_min, v_max):
    """Single Track Kinematic 모델(5-state: x,y,steer,v,yaw) — 원본과 동일.
    x = [x위치, y위치, 앞바퀴 조향각, 속도, yaw각]. u_init = [조향각속도, 가속도]."""
    lwb = lf + lr

    u = [steering_constraint(x[2], u_init[0], s_min, s_max, sv_min, sv_max),
         accl_constraints(x[3], u_init[1], v_switch, a_max, v_min, v_max)]

    f = [x[3] * math.cos(x[4]),
         x[3] * math.sin(x[4]),
         u[0],
         u[1],
         x[3] / lwb * math.tan(x[2])]
    return f


@njit(cache=True)
def vehicle_dynamics_st(x, u_init, mu, C_Sf, C_Sr, lf, lr, h, m, I,
                         s_min, s_max, sv_min, sv_max, v_switch, a_max, v_min, v_max):
    """Single Track Dynamic 모델(7-state: x,y,steer,v,yaw,yaw rate,slip) — 원본과 동일.
    x = [x위치, y위치, 앞바퀴 조향각, 속도, yaw각, yaw rate, 무게중심 슬립각].
    u_init = [조향각속도, 가속도]. 속도 절댓값이 0.5m/s 미만이면 내부적으로 KS 모델로
    자동 전환한다(원본 그대로 — ST 모델 자체가 v≈0에서 나눗셈 특이점을 가짐)."""
    g = 9.81

    u = [steering_constraint(x[2], u_init[0], s_min, s_max, sv_min, sv_max),
         accl_constraints(x[3], u_init[1], v_switch, a_max, v_min, v_max)]

    if abs(x[3]) < 0.5:
        lwb = lf + lr
        x_ks = x[0:5]
        f_ks = vehicle_dynamics_ks(x_ks, u, mu, C_Sf, C_Sr, lf, lr, h, m, I,
                                    s_min, s_max, sv_min, sv_max, v_switch, a_max, v_min, v_max)
        f = [f_ks[0], f_ks[1], f_ks[2], f_ks[3],
             f_ks[4],
             u[1] / lwb * math.tan(x[2]) + x[3] / (lwb * math.cos(x[2]) ** 2) * u[0],
             0.0]
    else:
        f = [x[3] * math.cos(x[6] + x[4]),
             x[3] * math.sin(x[6] + x[4]),
             u[0],
             u[1],
             x[5],
             -mu * m / (x[3] * I * (lr + lf)) * (lf ** 2 * C_Sf * (g * lr - u[1] * h) + lr ** 2 * C_Sr * (g * lf + u[1] * h)) * x[5]
                 + mu * m / (I * (lr + lf)) * (lr * C_Sr * (g * lf + u[1] * h) - lf * C_Sf * (g * lr - u[1] * h)) * x[6]
                 + mu * m / (I * (lr + lf)) * lf * C_Sf * (g * lr - u[1] * h) * x[2],
             (mu / (x[3] ** 2 * (lr + lf)) * (C_Sr * (g * lf + u[1] * h) * lr - C_Sf * (g * lr - u[1] * h) * lf) - 1) * x[5]
                 - mu / (x[3] * (lr + lf)) * (C_Sr * (g * lf + u[1] * h) + C_Sf * (g * lr - u[1] * h)) * x[6]
                 + mu / (x[3] * (lr + lf)) * (C_Sf * (g * lr - u[1] * h)) * x[2]]

    return f


@njit(cache=True)
def pid(speed, steer, current_speed, current_steer, max_sv, max_a, max_v, min_v):
    """desired speed/steer(각도) → accl./steer velocity 변환기 — 원본과 동일.
    vehicle_dynamics_st/ks가 요구하는 u_init(조향각속도, 가속도)는 우리
    PurePursuitController가 내놓는 절대 조향각/속도와 형식이 다른데, 이 함수가 그
    변환을 맡는다 — 동시에 max_sv/max_a가 실제 조향서보/모터의 반응속도 한계로
    작동해 "급조향 직후 진동" 같은 액추에이터 지연 효과가 시뮬레이션에 자연히
    반영된다(기존 기구학 모델은 조향/속도가 매 틱 즉시 명령대로 되는 것으로 가정했음)."""
    steer_diff = steer - current_steer
    if abs(steer_diff) > 1e-4:
        sv = (steer_diff / abs(steer_diff)) * max_sv
    else:
        sv = 0.0

    vel_diff = speed - current_speed
    if current_speed > 0.:
        if vel_diff > 0:
            kp = 10.0 * max_a / max_v
            accl = kp * vel_diff
        else:
            kp = 10.0 * max_a / (-min_v)
            accl = kp * vel_diff
    else:
        if vel_diff > 0:
            kp = 2.0 * max_a / max_v
            accl = kp * vel_diff
        else:
            kp = 2.0 * max_a / (-min_v)
            accl = kp * vel_diff

    return accl, sv
