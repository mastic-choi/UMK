#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#=============================================
# lqr.py — 차선 경로(카메라 ROI 픽셀 웨이포인트)를 LQR(선형 이차 조절기)로 추종하는
# 조향 컨트롤러. controller/pure_pursuit.py를 대체할 목적으로 같은 입출력 계약
# (control(path, vehicle_xy) -> 조향각(도), path 비면 직전값 유지)을 그대로 따른다 —
# track_drive.py 쪽 호출부는 어느 컨트롤러를 쓰든 수정할 필요가 없다.
#
# ── 모델: 왜 4-state(동역학) 대신 2-state(운동학) 오차모델인가 ──
#   Apollo 등에서 쓰는 LQR은 보통 [횡오차, 횡오차변화율, 헤딩오차, 헤딩오차변화율] 4개
#   상태에 코너링 강성(Cf,Cr), 질량, 요관성모멘트까지 넣는 동역학 모델을 쓴다. 이 차량은
#   그런 값이 실측된 적이 없고(축거조차 미실측 — controller/pure_pursuit.py 상단 주석
#   참고) 저속 소형차라 동역학 효과가 무시할 만한 수준이라, 그 복잡도를 감수할 이유가
#   없다. 대신 운동학 자전거모델을 선형화해 상태를 [횡오차 e_y, 헤딩오차 e_psi] 2개로
#   줄인다(속도 v, 축거 L, 조향각 delta):
#     e_y_dot   = v * sin(e_psi) ≈ v * e_psi        (소각근사)
#     e_psi_dot = v/L * delta                        (곡률 피드포워드는 v1에서 생략 — 아래 참고)
#
# ── v1 설계 결정: 곡률 피드포워드를 뺀 이유 ──
#   교과서/Apollo식 LQR은 delta = delta_ff(경로 곡률로 미리 꺾어두는 항) + delta_fb(-Kx)
#   구조라 곡선에서도 정상상태오차가 없다. 하지만 delta_ff 부호를 이미지 픽셀좌표계
#   (x=오른쪽+, "전방"=이미지 위쪽)에서 잘못 유도하면 코너에서 오히려 반대로 꺾어
#   상황을 악화시킬 위험이 있고, 그 오류는 저속 직선 구간 테스트로는 안 드러나고
#   실제 코너 진입 때만 나타난다 — 검증 안 된 부호 실수를 실차 코너 테스트로만 잡아야
#   하는 위험한 조합이라 v1에서는 뺐다. 순수 피드백(-Kx)만으로는 곡선 구간에서 약간의
#   정상상태 횡오차가 남을 수 있는데(직선 지그재그를 잡는 게 1차 목표이므로) 우선
#   이 상태로 실차 검증하고, 안정적으로 도는 게 확인되면 곡률 피드포워드를 별도
#   커밋으로 추가할 것.
#
# ── 좌표계: pure_pursuit.py와 동일한 이유로 미터가 아니라 픽셀 단위 ──
#   PIXELS_PER_METER가 아직 미실측(track_drive.py 참고)이라 e_y를 실제 미터로 못
#   바꾼다. 그래서 v(속도)도 "엔코더로 잰 실제 m/s"가 아니라 speed_gain이라는 튜닝
#   게인으로, L(축거)도 wheelbase_gain이라는 튜닝 게인으로 대신 쓴다 —
#   pure_pursuit.py가 축거 자리에 wheelbase_px 게인을 쓴 것과 똑같은 타협이다.
#   엔코더(localization/pose_estimator.py의 EncoderPoseEstimator)로 실측한 진짜
#   속도(m/s)를 쓰는 물리적으로 올바른 LQR로 전환하려면:
#     ① PIXELS_PER_METER 실측 → e_y를 미터로 변환
#     ② speed_gain 자리에 실제 속도(m/s) 대입 (매 프레임 갱신 필요 — 아래 set_speed 참고)
#     ③ wheelbase_gain 자리에 실측 축거(m) 대입
#   이 세 가지가 갖춰지기 전까지 아래 게인들은 물리적 의미가 없는 튜닝값이다.
#
# ── 부호 규약 (반드시 pure_pursuit.py와 통일) ──
#   이미지 x: 오른쪽으로 갈수록 +. 이미지 y: 아래로 갈수록 +(전방 = y가 작아지는 방향).
#   조향각(steer_deg): + = 우회전 (track_drive.py TURN_ANGLE=-60이 좌회전인 것과 일치).
#   e_y = vehicle_x - path_x  → 차량이 차선 중심보다 오른쪽에 있으면 +.
#   psi_path = atan2(dx, dy)  → 차량 정면 기준, 경로가 오른쪽을 향하면 +
#     (pure_pursuit._target_point()의 alpha 계산과 동일 공식).
#   e_psi(표준 정의, 차량헤딩-경로헤딩) = -psi_path. 유도: 차량이 실제로 차선보다
#     오른쪽을 보고 있으면(e_psi>0), 차량 시점(카메라)에는 차선이 왼쪽으로 휘어
#     보인다(운전 중 핸들을 우측으로 꺾으면 도로가 시야에서 왼쪽으로 쏠려 보이는 것과
#     같은 원리) → dx<0 → psi_path<0. 즉 둘은 항상 반대부호.
#   이 부호로 e_y>0, e_psi=0일 때 -Kx가 음수(좌회전)가 나와야 맞다 — K는 Q,R>0에서
#   나온 LQR 게인이라 항상 양수이므로 -K@[e_y,0] = -K1*e_y < 0, 정상.
#=============================================
import math

import numpy as np


class LQRController:
    """path(ROI 픽셀좌표 웨이포인트, 가까운점→먼점)와 차량 기준점을 받아 조향각(도)을
    반환한다. path가 비어있거나 너무 짧으면(min_path_px 미만) 직전 조향각을 유지한다 —
    pure_pursuit.PurePursuitController와 동일한 "못 믿을 프레임은 직전값 유지" 원칙."""

    # ── 기본값 출처 (2026-08-05, controller/pure_pursuit.py HEAD 기준) ──
    #   같은 픽셀 좌표계·같은 카메라/ROI를 보는 컨트롤러라 pure_pursuit.py가 이미
    #   실차에서 다듬어온 값들을 "1:1로 같은 역할"인 파라미터에는 그대로 물려받았다.
    #   그럴 만한 대응값이 아예 없는 파라미터(아래 표의 "대응 없음")는 정직하게 새
    #   기본값으로 뒀다 — pure_pursuit 공식엔 속도/시간/가중치 개념 자체가 없어서
    #   "가져올" 값이 없기 때문이다. 이 둘을 섞어서 전부 "실측 기반"인 것처럼 보이면
    #   안 되므로 아래에 표시해뒀다.
    #     wheelbase_gain=50.0     ← pure_pursuit.wheelbase_px=50.0        (동일 파라미터, 그대로 이식)
    #     angle_max_deg=100.0     ← pure_pursuit.angle_max_deg=100.0      (동일 파라미터, 그대로 이식)
    #     alpha=0.5               ← pure_pursuit.alpha=0.5                (동일 파라미터, 그대로 이식)
    #     heading_probe_px=65.0   ← pure_pursuit.min_lookahead_px=65.0    (다른 용도지만 같은 교훈 재사용:
    #                                짧은 거리에서 픽셀 노이즈가 각도로 증폭되는 문제를 그 팀원이 이미
    #                                실측으로 겪었던 값이라, 헤딩 추정 참조거리도 그보다 짧게 잡을 이유가 없음)
    #     min_path_px=65.0        ← pure_pursuit.min_lookahead_px=65.0    (위와 같은 이유로 재사용)
    #     speed_gain=120.0        ← 대응 없음. pure_pursuit 공식(atan(curvature*wheelbase_px))엔 애초에
    #                                속도/dt 개념이 없어서(기하학적, 시간과 무관) 물려받을 값이 없다.
    #                                순수 신규 튜닝값 — 실차에서 제일 먼저 건드려야 할 값이라고 보면 됨.
    #     q_lateral/q_heading/r_steer=1.0  ← 대응 없음. LQR 비용함수 가중치 자체가 pure_pursuit엔 없는
    #                                개념. 우선 1:1:1(균등)로 시작해서 실차 반응 보고 상대비율을 조정할 것
    #                                (예: 지그재그가 여전히 크면 r_steer를 올려 조향을 덜 공격적으로).
    #     dt=0.05                 ← track_drive.py control_loop 타이머 주기(20Hz)와 동일해야 하는 시스템
    #                                상수. "튜닝값"이 아니라 실제 주기와 반드시 맞춰야 하는 값.
    def __init__(self,
                 wheelbase_gain=50.0,
                 speed_gain=120.0,        # 대응 없음 — 신규 튜닝값(실차에서 최우선 튜닝 대상)
                 q_lateral=1.0,           # 대응 없음 — 신규 튜닝값. 횡오차(e_y) 가중치
                 q_heading=1.0,           # 대응 없음 — 신규 튜닝값. 헤딩오차(e_psi) 가중치
                 r_steer=1.0,             # 대응 없음 — 신규 튜닝값. 조향각 가중치(크게 잡을수록 덜 공격적)
                 dt=0.05,                 # 시스템 상수 — control_loop 타이머(20Hz)와 반드시 일치
                 heading_probe_px=65.0,   # ← pure_pursuit.min_lookahead_px
                 angle_max_deg=100.0,     # ← pure_pursuit.angle_max_deg
                 alpha=0.5,               # ← pure_pursuit.alpha
                 min_path_px=65.0):       # ← pure_pursuit.min_lookahead_px
        self.wheelbase_gain = wheelbase_gain
        self.speed_gain = speed_gain
        self.Q = np.diag([q_lateral, q_heading]).astype(np.float64)
        self.R = np.array([[r_steer]], dtype=np.float64)
        self.dt = dt
        self.heading_probe_px = heading_probe_px
        self.angle_max_deg = angle_max_deg
        self.alpha = alpha
        self.min_path_px = min_path_px

        self.prev_steer_deg = 0.0
        # A/B/K는 wheelbase_gain·speed_gain·dt가 고정인 한 상수이므로 매 프레임 Riccati를
        # 다시 풀 필요가 없다 — 최초 1회(혹은 게인이 바뀐 프레임)만 계산해 캐싱한다.
        self._cached_gain_key = None
        self._cached_K = None

        # 직전 control() 호출이 "새로 계산"했는지 "직전값을 그대로 유지"했는지, 그리고
        # 그때 쓴 오차상태값 — track_drive.py의 DEBUG_VIZ_STEER 디버그 창이 읽어서 보여준다.
        self.held = False
        self.last_e_y = None
        self.last_e_psi = None

    def reset(self):
        self.prev_steer_deg = 0.0
        self.held = False

    def set_speed_gain(self, speed_gain):
        """실측 속도(엔코더 등)를 연동하게 되면 매 프레임 이걸로 speed_gain을 갱신하고
        control()을 호출하면 된다 — A/B가 바뀌므로 다음 control() 호출 때 K가 자동으로
        재계산된다(캐시 키에 speed_gain이 들어있음)."""
        self.speed_gain = speed_gain

    # ── 이산시간 LQR 게인 계산 (Riccati 방정식을 직접 반복수렴) ──
    #   scipy.linalg.solve_discrete_are를 안 쓰는 이유: 이 저장소에서 scipy는
    #   perc_lavacon.py(Voronoi)에서만 쓰이고 실차 온보드에 항상 설치돼 있다는 보장이
    #   없다(이 세션 sandbox에도 미설치로 확인됨). 2x2 상태라 반복수렴이 훨씬 가볍고
    #   외부 의존성도 늘지 않는다.
    def _solve_dlqr(self, A, B, Q, R, max_iter=200, tol=1e-9):
        P = Q.copy()
        for _ in range(max_iter):
            BtP = B.T @ P
            S = R + BtP @ B
            K = np.linalg.solve(S, BtP @ A)
            P_next = Q + A.T @ P @ A - A.T @ P @ B @ K
            if np.max(np.abs(P_next - P)) < tol:
                P = P_next
                break
            P = P_next
        BtP = B.T @ P
        K = np.linalg.solve(R + BtP @ B, BtP @ A)
        return K

    def _gain(self):
        key = (self.wheelbase_gain, self.speed_gain, self.dt)
        if self._cached_gain_key == key:
            return self._cached_K
        v = self.speed_gain
        L = self.wheelbase_gain
        A = np.array([[1.0, v * self.dt],
                      [0.0, 1.0]], dtype=np.float64)
        B = np.array([[0.0],
                      [v * self.dt / L]], dtype=np.float64)
        K = self._solve_dlqr(A, B, self.Q, self.R)
        self._cached_gain_key = key
        self._cached_K = K
        return K

    def _path_len(self, path):
        return sum(
            math.hypot(path[i][0] - path[i - 1][0], path[i][1] - path[i - 1][1])
            for i in range(1, len(path))
        )

    def _heading_probe_point(self, path):
        """누적호길이가 heading_probe_px를 넘는 첫 지점의 인덱스. 경로가 그보다
        짧으면(짧은 ROI 등) path의 마지막 점을 쓴다 — pure_pursuit._target_point()와
        동일한 폴백 원칙."""
        acc = 0.0
        for i in range(1, len(path)):
            acc += math.hypot(path[i][0] - path[i - 1][0], path[i][1] - path[i - 1][1])
            if acc >= self.heading_probe_px:
                return i
        return len(path) - 1

    def control(self, path, vehicle_xy):
        """반환 : 조향각(도). 호출 후 self.held로 "새로 계산"(False)했는지 "직전값
        유지"(True)했는지, self.last_e_y/last_e_psi로 이번(혹은 마지막 유효) 오차상태값을
        확인할 수 있다(디버그 창용)."""
        if not path or len(path) < 2:
            self.held = True
            return self.prev_steer_deg
        if self._path_len(path) < self.min_path_px:
            self.held = True
            return self.prev_steer_deg

        # ── 오차상태 계산 (부호 규약은 파일 상단 주석 참고) ──
        e_y = vehicle_xy[0] - path[0][0]

        probe_i = self._heading_probe_point(path)
        px, py = path[probe_i]
        dx = px - vehicle_xy[0]
        dy = max(vehicle_xy[1] - py, 1e-6)
        psi_path = math.atan2(dx, dy)
        e_psi = -psi_path
        self.last_e_y = e_y
        self.last_e_psi = e_psi

        K = self._gain()
        x = np.array([[e_y], [e_psi]], dtype=np.float64)
        delta_rad = float(-(K @ x)[0, 0])

        steer_deg = math.degrees(delta_rad)
        steer_deg = self.alpha * steer_deg + (1.0 - self.alpha) * self.prev_steer_deg
        steer_deg = float(np.clip(steer_deg, -self.angle_max_deg, self.angle_max_deg))
        self.prev_steer_deg = steer_deg
        self.held = False
        return steer_deg
