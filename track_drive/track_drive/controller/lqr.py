#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#=============================================
# lqr.py — 차선 경로(카메라 ROI 픽셀 웨이포인트)를 LQR(선형 이차 조절기)로 추종하는
# 조향 컨트롤러. controller/pure_pursuit.py를 대체할 목적으로 같은 입출력 계약
# (control(path, vehicle_xy) -> 조향각(도), path 비면 직전값 유지)을 그대로 따른다 —
# track_drive.py 쪽 호출부는 어느 컨트롤러를 쓰든 수정할 필요가 없다.
#
# ── ★ 2026-08-05 실차 영상에서 발견된 버그와 수정 내역 (LQR 브랜치에서 이식) ★ ──
#   BEV 적용 후 실차 영상(steer_debug 창)에서 e_y=-7.7px, e_psi=-4.4도 같은 사실상
#   미세한 오차에도 조향각이 +100도(클램프 최대치)로 튀는 게 확인됨. 원인은 BEV가
#   아니라 애초에 있던 단위 버그였다: Q=diag(1,1)이 e_y(픽셀, 크기 O(1~100))와
#   e_psi(라디안, 크기 O(0.01~0.5))를 "같은 가중치 1"로 취급하면서, Riccati가 e_y를
#   억누르는 데 몰빵한 극단적으로 큰 K를 내놓고 있었다(검산: 기존 기본값으로
#   e_y=-40px 하나만 넣어도 클램프 전 raw 조향각이 1234도가 나옴 — 애초에 BEV 이전
#   원근 좌표계에서도 똑같이 터졌어야 하는 값인데, 그땐 raw offset이 우연히 크게
#   나오는 경우가 많아 알아채기 어려웠을 뿐).
#   BEV 덕분에 이제 DL_PIXELS_PER_METER(config.py, =200px/m, 실측 4점 기반 설계값이라
#   정확함)라는 "진짜" 픽셀↔미터 환산이 생겼으므로, 이 파일도 드디어 물리적으로 올바른
#   단위(e_y를 미터로 환산, 실제 축거/속도 사용)로 LQR을 풀 수 있다 — 아래
#   pixels_per_meter 파라미터가 그 스위치다. 켜면 e_y·e_psi가 둘 다 O(0.01~0.5) 정도의
#   비슷한 크기가 되어 Q=diag(1,1)이 비로소 "말이 되는" 균등가중이 된다(검산: 같은
#   e_y=-40px에서 raw 조향각이 1234도 → 10.2도로 정상화됨).
#
# ── 모델: 왜 4-state(동역학) 대신 2-state(운동학) 오차모델인가 ──
#   Apollo 등에서 쓰는 LQR은 보통 [횡오차, 횡오차변화율, 헤딩오차, 헤딩오차변화율] 4개
#   상태에 코너링 강성(Cf,Cr), 질량, 요관성모멘트까지 넣는 동역학 모델을 쓴다. 이 차량은
#   그런 값이 실측된 적이 없고 저속 소형차라 동역학 효과가 무시할 만한 수준이라, 그
#   복잡도를 감수할 이유가 없다. 대신 운동학 자전거모델을 선형화해 상태를 [횡오차 e_y,
#   헤딩오차 e_psi] 2개로 줄인다(속도 v, 축거 L, 조향각 delta):
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
# ── 좌표계: 픽셀 vs 미터 두 모드를 다 지원 ──
#   pixels_per_meter를 주면(예: config.py의 DL_PIXELS_PER_METER=200.0 — BEV 워프
#   목적캔버스를 1m=200px로 "설계"했으므로 근사가 아니라 정확한 값) 내부적으로 e_y를
#   미터로 환산하고, wheelbase_m(실측 축거, m)·speed_mps(추정/실측 속도, m/s)로
#   물리적으로 올바른 LQR을 푼다.
#   pixels_per_meter를 안 주면(None, 기본값) 예전처럼 e_y를 픽셀 그대로 쓰고
#   wheelbase_gain/speed_gain을 "물리적 의미 없는 튜닝 게인"으로 쓰는 레거시 모드로
#   동작한다 — BEV 워프가 없는 백엔드(hough, 혹은 DL_USE_BEV=False인 dl)에서 픽셀↔미터
#   환산을 모를 때 쓰는 폴백이다. 어느 모드든 Q/R 가중치는 "e_y와 e_psi가 비슷한
#   크기 단위인지"를 항상 먼저 확인할 것 — 이번 버그가 정확히 그 확인을 안 해서 생겼다.
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
    반환한다. path가 비어있거나 너무 짧으면 직전 조향각을 유지한다 —
    pure_pursuit.PurePursuitController와 동일한 "못 믿을 프레임은 직전값 유지" 원칙."""

    def __init__(self,
                 pixels_per_meter=None,   # BEV 등에서 px<->m 환산을 알면 그 값(px/m). None=레거시 픽셀 모드.
                 wheelbase_m=0.335,       # 실측값(2026-08-06, 줄자 실측 — LQR 브랜치에서 이식).
                                          #   pixels_per_meter가 있을 때만 사용.
                 speed_mps=1.0,           # 주행 속도 추정치(m/s). 엔코더 실측 전 임시값 — 실차에서 최우선
                                          #   튜닝 대상. localization.EncoderPoseEstimator 연결되면 그 값으로
                                          #   매 프레임 set_speed_mps() 갱신할 것. pixels_per_meter가 있을 때만 사용.
                 wheelbase_gain=50.0,     # [레거시 모드 전용] pixels_per_meter=None일 때만 사용.
                 speed_gain=120.0,        # [레거시 모드 전용] 위와 동일.
                 q_lateral=1.0,           # Q 대각원소: 횡오차(e_y) 가중치.
                 q_heading=1.0,           # Q 대각원소: 헤딩오차(e_psi) 가중치.
                 r_steer=1.0,             # R(스칼라): 조향각 가중치 — 크게 잡을수록 덜 공격적(진동 억제 방향).
                 dt=0.05,                 # 시스템 상수 — control_loop 타이머(20Hz)와 반드시 일치.
                 heading_probe_m=0.3,     # [미터 모드] 헤딩오차 추정용 근거리 참조거리(m).
                 min_path_m=0.3,          # [미터 모드] 경로 전체 길이가 이보다 짧으면 직전값 유지.
                 heading_probe_px=65.0,   # [레거시 모드] ← pure_pursuit.min_lookahead_px와 동일값 이식.
                 min_path_px=65.0,        # [레거시 모드] ← pure_pursuit.min_lookahead_px와 동일값 이식.
                 angle_max_deg=100.0,
                 alpha=0.5):
        self.pixels_per_meter = pixels_per_meter

        if pixels_per_meter:
            # ── 미터 모드: 물리적으로 올바른 값 사용 ──
            self._L = wheelbase_m
            self._v = speed_mps
            self.heading_probe_px = heading_probe_m * pixels_per_meter
            self.min_path_px = min_path_m * pixels_per_meter
        else:
            # ── 레거시 픽셀 모드: 튜닝 게인 사용(물리적 의미 없음) ──
            self._L = wheelbase_gain
            self._v = speed_gain
            self.heading_probe_px = heading_probe_px
            self.min_path_px = min_path_px

        self.Q = np.diag([q_lateral, q_heading]).astype(np.float64)
        self.R = np.array([[r_steer]], dtype=np.float64)
        self.dt = dt
        self.angle_max_deg = angle_max_deg
        self.alpha = alpha

        self.prev_steer_deg = 0.0
        # A/B/K는 (L,v,dt)가 고정인 한 상수이므로 매 프레임 Riccati를 다시 풀 필요가
        # 없다 — 최초 1회(혹은 게인이 바뀐 프레임)만 계산해 캐싱한다.
        self._cached_gain_key = None
        self._cached_K = None

        # 직전 control() 호출이 "새로 계산"했는지 "직전값을 그대로 유지"했는지, 그리고
        # 그때 쓴 오차상태값(디버그 창 표시는 항상 픽셀 단위로 통일) —
        # track_drive.py의 DEBUG_VIZ_STEER 디버그 창이 읽어서 보여준다.
        self.held = False
        self.last_e_y = None
        self.last_e_psi = None

    def reset(self):
        self.prev_steer_deg = 0.0
        self.held = False

    def set_speed_mps(self, speed_mps):
        """[미터 모드] 엔코더 등으로 실측한 속도(m/s)를 매 프레임 갱신할 때 사용.
        A/B가 바뀌므로 다음 control() 호출 때 K가 자동으로 재계산된다."""
        self._v = speed_mps

    def set_speed_gain(self, speed_gain):
        """[레거시 모드] 이전과 동일한 이름으로 유지(하위호환). 미터 모드에서는
        set_speed_mps()를 쓸 것 — pixels_per_meter가 설정돼 있으면 이 값도 그대로
        self._v에 들어가긴 하지만 단위가 섞이므로 미터 모드에서는 쓰지 말 것."""
        self._v = speed_gain

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
        key = (self._L, self._v, self.dt)
        if self._cached_gain_key == key:
            return self._cached_K
        v = self._v
        L = self._L
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
        유지"(True)했는지, self.last_e_y(px)/last_e_psi(rad)로 이번(혹은 마지막 유효)
        오차상태값을 확인할 수 있다(디버그 창용, 항상 픽셀 단위로 표시)."""
        if not path or len(path) < 2:
            self.held = True
            return self.prev_steer_deg
        if self._path_len(path) < self.min_path_px:
            self.held = True
            return self.prev_steer_deg

        # ── 오차상태 계산 (부호 규약은 파일 상단 주석 참고, 항상 픽셀 단위로 먼저 계산) ──
        e_y_px = vehicle_xy[0] - path[0][0]

        probe_i = self._heading_probe_point(path)
        px, py = path[probe_i]
        dx = px - vehicle_xy[0]
        dy = max(vehicle_xy[1] - py, 1e-6)
        psi_path = math.atan2(dx, dy)
        e_psi = -psi_path
        self.last_e_y = e_y_px
        self.last_e_psi = e_psi

        # 미터 모드면 LQR 계산 직전에만 e_y를 미터로 환산(디버그 표시는 위에서 이미
        # 픽셀로 저장해뒀으니 영향 없음) — e_psi는 이미 라디안(단위 그대로 미터 모드와
        # 호환)이라 변환 불필요.
        e_y = e_y_px / self.pixels_per_meter if self.pixels_per_meter else e_y_px

        K = self._gain()
        x = np.array([[e_y], [e_psi]], dtype=np.float64)
        delta_rad = float(-(K @ x)[0, 0])

        steer_deg = math.degrees(delta_rad)
        steer_deg = self.alpha * steer_deg + (1.0 - self.alpha) * self.prev_steer_deg
        steer_deg = float(np.clip(steer_deg, -self.angle_max_deg, self.angle_max_deg))
        self.prev_steer_deg = steer_deg
        self.held = False
        return steer_deg
