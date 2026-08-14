#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#=============================================
# steering_stability_lib.py — 조향 안정성 개선 후보 알고리즘 구현체.
#
# steering_stability.md(§2~§5)에서 설계한 4개 기법을 코드로 옮겨놓은 것이다.
#
# ★★★ 아직 실제 주행 파이프라인(perception/lane_util.py, controller/pure_pursuit.py,
#     track_drive.py) 어디에서도 import되지 않는다 — 검토/단위테스트 전용 코드다. ★★★
# 실제로 적용하려면 steering_stability.md §8 "권장 진행 순서"를 참고해 해당 파일들을
# 직접 수정해서 여기 있는 클래스/함수를 갖다 써야 한다(이 파일 자체를 고치는 게 아니라
# import해서 쓰는 방식을 권장 — 기존 파일 쪽 변경은 최소화하고 롤백도 쉬워짐).
#
# 포함된 것:
#   1) WaypointKalmanFilter — 경로 스무딩용 등속도(위치+속도) 1D 칼만필터 (§2)
#   2) SlewRateLimiter      — 조향각 출력 하드 캡 (§3)
#   3) uncertainty_speed_scale — 칼만필터 공분산 → 속도 배율 (§4)
#   4) ComplementaryFilter  — 비전+IMU offset 융합 (§5)
#=============================================
import numpy as np


# ─────────────────────────────────────────────
# §2 — 경로 스무딩: 등속도(위치+속도) 1D 칼만필터
# ─────────────────────────────────────────────
def _kalman_step(z, x, vx, p00, p01, p11, q_pos, q_vel, r):
    """등속도(위치+속도) 1D 칼만필터 한 스텝(예측→갱신)을 트랙 N개에 한번에 적용한다
    (트랙끼리 서로 독립이라 전부 원소별(elementwise) numpy 연산으로 벡터화했다 — 결과는
    트랙마다 따로 2x2 행렬 KF를 돌리는 것과 수학적으로 동일하다).
      상태전이 F=[[1,1],[0,1]](dt=1프레임 등속도 가정), 프로세스노이즈 Q=diag(q_pos,q_vel),
      측정모델 H=[1,0](위치만 관측), 측정노이즈 r. 공분산 P의 대칭성을 이용해 고유항
      3개(p00,p01,p11)만 추적한다(p10=p01이므로 별도 보관 불필요).
      입력 : z(측정값)/x,vx(직전 상태)/p00,p01,p11(직전 공분산) — 전부 같은 shape의 배열,
             q_pos/q_vel/r — 스칼라
      출력 : (x, vx, p00, p01, p11) — 갱신된 상태/공분산(입력과 같은 shape)
    """
    # 예측: x_pred = F @ x, P_pred = F @ P @ F.T + Q
    x_pred = x + vx
    vx_pred = vx
    p00_pred = p00 + 2.0 * p01 + p11 + q_pos
    p01_pred = p01 + p11
    p11_pred = p11 + q_vel

    # 갱신: H=[1,0]이라 위치만 관측. K = P_pred H^T / (H P_pred H^T + r)
    innovation = z - x_pred
    s = p00_pred + r
    k0 = p00_pred / s
    k1 = p01_pred / s

    x_new = x_pred + k0 * innovation
    vx_new = vx_pred + k1 * innovation
    # P_new = P_pred - K H P_pred (H P_pred = [p00_pred, p01_pred] 행)
    p00_new = (1.0 - k0) * p00_pred
    p01_new = (1.0 - k0) * p01_pred
    p11_new = p11_pred - k1 * p01_pred

    return x_new, vx_new, p00_new, p01_new, p11_new


class WaypointKalmanFilter:
    """경로(웨이포인트 리스트)를 프레임 간 스무딩하는 등속도 칼만필터.

    perception/lane_util.py의 SlideWindow._update_path()(현재 EMA 방식)를 대체할
    용도로 설계됐다 — 인터페이스를 그와 최대한 맞춰서(update(fitted_path) -> path)
    나중에 실제 파일에 옮겨 붙이기 쉽게 했다.

    사용 예:
        kf = WaypointKalmanFilter(q_pos=4.0, q_vel=1.0, r=64.0, init_var=10000.0)
        smoothed_path = kf.update(fitted_path)   # fitted_path: [(x,y), ...] 또는 None
        uncertainty = kf.mean_position_variance   # §4가 쓰는 값
    """

    def __init__(self, q_pos=4.0, q_vel=1.0, r=64.0, init_var=10000.0):
        self.q_pos = q_pos
        self.q_vel = q_vel
        self.r = r
        self.init_var = init_var

        self._x = None
        self._vx = None
        self._p00x = None
        self._p01x = None
        self._p11x = None
        self._y = None
        self._vy = None
        self._p00y = None
        self._p01y = None
        self._p11y = None

        self.path = []

    def reset(self, fitted_path):
        """웨이포인트별 트랙을 fitted_path 값으로 (재)초기화한다 — 속도는 0, 공분산은
        init_var(크게 잡아 첫 갱신에서 실측값을 거의 그대로 신뢰하게 함)."""
        n = len(fitted_path)
        xs = np.array([p[0] for p in fitted_path], dtype=np.float64)
        ys = np.array([p[1] for p in fitted_path], dtype=np.float64)

        self._x, self._vx = xs, np.zeros(n)
        self._y, self._vy = ys, np.zeros(n)
        self._p00x = np.full(n, self.init_var)
        self._p01x = np.zeros(n)
        self._p11x = np.full(n, self.init_var)
        self._p00y = np.full(n, self.init_var)
        self._p01y = np.zeros(n)
        self._p11y = np.full(n, self.init_var)

        self.path = fitted_path

    def update(self, fitted_path):
        """새로 피팅된 경로(fitted_path)를 측정값으로 넣어 필터를 갱신하고, 필터링된
        경로를 반환한다(self.path에도 저장).

        fitted_path가 None이면(유효 슬라이스 부족 등 무효 프레임) predict 단계조차
        건너뛰고 완전히 정지한다 — "무효 프레임엔 마지막 값 유지" 원칙(무효 프레임이
        여러 프레임 이어질 때 속도 상태만으로 예측이 계속 벌어지는 걸 방지).

        직전 경로가 없거나(첫 호출) 웨이포인트 개수가 바뀌면 reset()으로 트랙을
        재초기화하고 새 경로를 그대로 채택한다.
        """
        if fitted_path is None:
            return self.path

        if self._x is None or len(self.path) != len(fitted_path):
            self.reset(fitted_path)
            return self.path

        zx = np.array([p[0] for p in fitted_path], dtype=np.float64)
        zy = np.array([p[1] for p in fitted_path], dtype=np.float64)

        self._x, self._vx, self._p00x, self._p01x, self._p11x = _kalman_step(
            zx, self._x, self._vx, self._p00x, self._p01x, self._p11x,
            self.q_pos, self.q_vel, self.r,
        )
        self._y, self._vy, self._p00y, self._p01y, self._p11y = _kalman_step(
            zy, self._y, self._vy, self._p00y, self._p01y, self._p11y,
            self.q_pos, self.q_vel, self.r,
        )
        self.path = list(zip(self._x.tolist(), self._y.tolist()))
        return self.path

    @property
    def mean_position_variance(self):
        """웨이포인트 전체의 평균 위치 불확실성(x축 공분산 p00 기준) — §4가 속도 계획에
        연동하는 값. 아직 한 번도 갱신 안 됐으면(초기화 전) 0.0."""
        if self._p00x is None:
            return 0.0
        return float(np.mean(self._p00x))


# ─────────────────────────────────────────────
# §3 — 조향각 출력 슬루레이트 리미터
# ─────────────────────────────────────────────
class SlewRateLimiter:
    """출력이 "한 스텝에 최대 이만큼까지만" 바뀌도록 절대 상한을 씌우는 필터.
    EMA(비율 기반 저역통과)와 달리, 입력이 아무리 극단적으로 튀어도 한 스텝에 정해진
    변화량 이상은 절대 못 바뀐다는 확정적(deterministic) 보장을 준다.

    controller/pure_pursuit.py의 self.alpha EMA 블렌딩(steer_deg = alpha*raw +
    (1-alpha)*prev)을 대체할 용도로 설계됐다.

    사용 예:
        limiter = SlewRateLimiter(max_step=5.0)
        steer_deg = limiter.step(raw_steer_deg)
    """

    def __init__(self, max_step, initial_value=0.0):
        self.max_step = max_step
        self.prev_value = initial_value

    def step(self, raw_value):
        diff = raw_value - self.prev_value
        diff = float(np.clip(diff, -self.max_step, self.max_step))
        self.prev_value = self.prev_value + diff
        return self.prev_value

    def reset(self, value=0.0):
        self.prev_value = value


# ─────────────────────────────────────────────
# §4 — 칼만필터 공분산 → 속도 감속 연동
# ─────────────────────────────────────────────
def uncertainty_speed_scale(path_uncertainty, gain, min_scale):
    """WaypointKalmanFilter.mean_position_variance(§2) 값을 받아 속도 배율(0~1)을
    반환한다 — 인식이 불확실할수록(공분산이 클수록) 배율이 낮아져 자동 감속된다.
    기존 _corner_radius_speed_scale()(회전반경 기반 감속)과 같은 철학의 함수.

      target_speed *= uncertainty_speed_scale(kf.mean_position_variance, K, MIN_SCALE)

    gain(K)이 클수록 같은 불확실성에도 더 세게 감속한다. min_scale은 완전히 멈추지
    않도록 하는 하한(예: 0.3 = 최소 30% 속도는 유지).
    """
    scale = 1.0 - gain * path_uncertainty
    return float(np.clip(scale, min_scale, 1.0))


# ─────────────────────────────────────────────
# §5 — IMU-비전 상보 필터 (offset 융합)
# ─────────────────────────────────────────────
class ComplementaryFilter:
    """서로 다른 두 신호(장기적으로 정확하지만 노이즈 있는 신호 A, 순간 반응은 빠르지만
    드리프트하는 신호 B의 변화량)를 섞는 필터. 이 프로젝트 맥락에서는:
      A = 비전 기반 lane_offset(느리지만 절대적으로 정확)
      B = IMU yaw_rate 적분값(빠르지만 계속 쌓으면 오차 누적)

    사용 예 (매 제어 틱 20Hz):
        cf = ComplementaryFilter(alpha=0.98)  # alpha가 클수록 비전(A)을 더 신뢰
        offset_filtered = cf.update(vision_offset=lane_offset,
                                     gyro_rate=imu_yaw_rate, dt=0.05,
                                     rate_to_offset_gain=GAIN)
    """

    def __init__(self, alpha=0.98):
        """alpha : 이번 스텝 추정치 중 "예측(IMU 적분)"에 줄 가중치가 아니라, 반대로
        "비전 측정치"에 줄 최종 신뢰 가중치로 뒀다 — 표준 상보필터 공식
        (output = alpha*(prev+gyro_integral) + (1-alpha)*vision)에서 alpha가 크면
        IMU(고주파) 비중이 커 보이지만, 시간이 지나면 결국 비전이 계속 끌어당기므로
        저주파(정상상태)는 항상 비전이 이긴다 — 이름의 "얼마나 IMU 예측을 오래 유지할지"
        정도로 이해할 것. 실차 미검증 첫 추정치."""
        self.alpha = alpha
        self.value = 0.0
        self._initialized = False

    def reset(self, value=0.0):
        self.value = value
        self._initialized = True

    def update(self, vision_offset, gyro_rate, dt, rate_to_offset_gain=1.0):
        """vision_offset : 이번 프레임 비전 기반 offset 측정치(없으면 None — 그 경우
                            IMU 예측만으로 한 스텝 진행하고 반환)
           gyro_rate      : IMU 각속도(yaw_rate, rad/s 등 프로젝트 단위)
           dt             : 이전 업데이트 이후 경과 시간(초)
           rate_to_offset_gain : gyro_rate(각속도)를 offset(픽셀 등 위치 단위) 변화량으로
                            환산하는 게인 — 차량 속도/기하에 따라 달라지므로 실측 필요.
        """
        if not self._initialized:
            self.reset(vision_offset if vision_offset is not None else 0.0)
            return self.value

        predicted = self.value + gyro_rate * rate_to_offset_gain * dt

        if vision_offset is None:
            self.value = predicted
        else:
            self.value = self.alpha * predicted + (1.0 - self.alpha) * vision_offset

        return self.value


if __name__ == '__main__':
    # 간단 자가 테스트(ROS 없이 로직 검증용) — 노이즈 섞인 직선 경로에 각 필터를 돌려
    # 그럴듯하게 스무딩되는지 눈으로 확인.
    rng = np.random.default_rng(0)

    print('--- WaypointKalmanFilter ---')
    kf = WaypointKalmanFilter()
    base_path = [(320.0 + i * 2.0, 400.0 - i * 30.0) for i in range(6)]
    for t in range(5):
        noisy = [(x + rng.normal(0, 8), y) for x, y in base_path]
        smoothed = kf.update(noisy)
        print(f'frame {t}: raw_x0={noisy[0][0]:.1f} smoothed_x0={smoothed[0][0]:.1f} '
              f'uncertainty={kf.mean_position_variance:.2f}')

    print('--- SlewRateLimiter ---')
    limiter = SlewRateLimiter(max_step=5.0)
    for raw in [0.0, 40.0, 40.0, 40.0, -30.0]:
        print(f'raw={raw:+.1f} -> limited={limiter.step(raw):+.2f}')

    print('--- uncertainty_speed_scale ---')
    for u in [0.0, 5.0, 20.0, 100.0]:
        print(f'uncertainty={u:.1f} -> speed_scale={uncertainty_speed_scale(u, gain=0.02, min_scale=0.3):.2f}')

    print('--- ComplementaryFilter ---')
    cf = ComplementaryFilter(alpha=0.9)
    cf.reset(0.0)
    for t in range(5):
        v = 10.0 + rng.normal(0, 2) if t % 2 == 0 else None  # 비전이 격프레임만 들어온다고 가정
        out = cf.update(vision_offset=v, gyro_rate=0.5, dt=0.05, rate_to_offset_gain=20.0)
        print(f'frame {t}: vision={v} -> fused={out:.2f}')
