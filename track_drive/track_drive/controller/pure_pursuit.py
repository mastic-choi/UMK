#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#=============================================
# pure_pursuit.py — DL/classic 세그멘테이션이 만든 BEV/ROI 픽셀 경로를 따라가는
# 기하학적(geometric) 조향 컨트롤러.
#
# 기존 방식(lane_util.SlideWindow가 근거리/원거리 두 그룹으로 뭉갠 offset/lookahead
# 스칼라 두 개를 track_drive.py의 PID(_lane_pid)에 먹이는 방식)과 달리, 이제
# lane_util._fit_and_sample_path()가 슬라이스별 차선 중심점에 다항식을 피팅해 만든
# 명시적 곡선(웨이포인트 리스트, ROI 픽셀좌표, 가까운점→먼점 순)을 그대로 받아
# Pure Pursuit 기하 공식으로 조향각을 계산한다.
#
# ── 좌표계: 왜 미터가 아니라 픽셀인가 ──
#   controller/stanley.py는 planner/hybrid_astar.py가 만든 실세계 미터 단위 경로를
#   차량 위치(x,y,yaw)와 함께 추종한다. 여기서는 그게 불가능한데, 카메라 픽셀→미터
#   변환(track_drive.py의 PIXELS_PER_METER)이 아직 실측 전(0.0)이기 때문이다. 그래서
#   차량은 항상 ROI 하단 중앙(vehicle_xy, path[0] 근방)에 고정된 것으로 보고, 카메라가
#   보는 위쪽 방향(-y)을 "전방"으로 삼아 픽셀 좌표계 안에서 그대로 기하 계산을 한다.
#   PIXELS_PER_METER가 실측되면 wheelbase_px를 "실제 축거리(m) * PIXELS_PER_METER"로
#   대체해 진짜 물리 단위 Pure Pursuit으로 전환할 수 있다.
#=============================================
import math

import numpy as np


class PurePursuitController:
    """path(ROI 픽셀좌표 웨이포인트, 가까운점→먼점)와 차량 기준점을 받아 조향각(도)을
    반환한다. path가 비어있으면(첫 프레임 등) 직전 조향각을 그대로 유지한다 —
    lane_util._debounce()가 무효 프레임에 직전 확정값을 유지하는 것과 같은 원칙."""

    def __init__(self, lookahead_base_px=90.0, lookahead_speed_gain=4.0, lookahead_max_px=150.0,
                 wheelbase_px=80.0, angle_max_deg=100.0, alpha=0.5,
                 min_lookahead_px=90.0, dx_deadzone_px=6.0):
        # [2026-08-05, 속도 적응형 lookahead — 1차 시도 후 수정]
        #   curvature ≈ 2*dx/ld^2 (작은각 근사)라서 ld가 짧을수록 같은 dx도 1/ld^2로
        #   증폭된다. 처음엔 PythonRobotics/Autoware 관례(Ld=k*v+Lfc, 저속일수록 lookahead도
        #   짧게)를 그대로 따라 lookahead_base_px를 65(=기존 min_lookahead_px 가드값과 동일)로
        #   낮췄는데, 실차에서 "waypoint가 살짝만 틀어져도(dx≈50px) ang=50도 가까이 찍힌다"는
        #   증상으로 재현됨 — ld=65, dx=50이면 steer≈50°(계산 확인됨), ld=90(구 고정값)이면
        #   같은 dx에도 steer≈32°로 훨씬 덜 민감하다. 즉 "저속=깨끗한 경로를 촘촘히 추종"을
        #   가정하는 PythonRobotics류 관례가, 매 프레임 픽셀 노이즈가 있는 비전 기반 경로에는
        #   그대로 안 맞았다(짧은 lookahead가 노이즈까지 증폭). 게다가 base와 min_lookahead_px
        #   가드가 같은 값(65)이라 이 위험구간에서 가드도 사실상 작동을 안 했다.
        #   → 하한을 기존에 검증된 고정값(90)으로 되돌리고, 속도가 오르면 그 "위로만" 늘어나게
        #   바꿨다(절대 90 밑으로는 안 내려감). 여전히 저속에서 tight-tracking 이점은 없지만,
        #   노이즈 증폭 문제가 없는 쪽을 우선했다 — 실차에서 저속 코너 추종이 둔하게 느껴지면
        #   base를 낮추기보다 wheelbase_px를 낮추는 쪽으로 대응할 것(하단 주석 참고).
        #   gain=4.0: SPEED_NORMAL(5.0)에서 90+4*5=110, 향후 최고속도를 올려도 max_px(150)
        #   여유 안에 들어오도록 잡은 것. 전부 실차 미검증 튜닝값.
        self.lookahead_base_px = lookahead_base_px
        self.lookahead_speed_gain = lookahead_speed_gain
        self.lookahead_max_px = lookahead_max_px

        # [2026-08-05] dx(waypoint-차량중앙 픽셀오차)가 이 값 미만이면 0으로 죽인다 —
        #   차량이 차선 중앙에 사실상 있는데도(카메라/세그멘테이션의 서브픽셀 단위 흔들림) 매
        #   프레임 미세하게 다른 조향이 나가 중앙 부근에서 계속 잔떨림하는 것을 막는다.
        #   _lane_pid()의 LANE_DEADZONE(40px, PID 입력용)과 같은 철학이지만, 여긴 훨씬 작은
        #   값을 쓴다 — Pure Pursuit은 목표점 자체가 이미 lookahead 앞의 실제 경로점이라
        #   LANE_DEADZONE만큼 크게 죽이면 완만한 커브 진입까지 무시하게 된다. 실차 미검증.
        self.dx_deadzone_px = dx_deadzone_px

        # 실제 차축거리(m) 대신 쓰는 "곡률→조향각" 게인. 표준 Pure Pursuit 공식
        # steer = atan(2*L*sin(alpha)/Ld)에서 L 자리에 들어간다 — 크게 잡을수록 같은
        # 곡률에도 조향각이 커진다(더 공격적). PIXELS_PER_METER 실측 전까지는 물리적
        # 의미가 없는 순수 튜닝 게인이므로, 실차에서 직진 안정성/코너 추종성을 보고
        # 맞출 것. 실차 미검증 튜닝값.
        #   [2026-08-05] lookahead 하한(90) 복원 후에도 여전히 작은 dx에 조향이 과민하면,
        #   lookahead보다 이 값을 낮추는 쪽이 더 직접적인 레버다 — curvature*wheelbase_px에
        #   그대로 곱해지므로 ld 범위 전체에서 균일하게 민감도를 낮춘다(단, 실제 코너에서도
        #   반응이 약해지니 함께 재확인할 것).
        self.wheelbase_px = wheelbase_px

        self.angle_max_deg = angle_max_deg

        # 목표점까지 남은 실제거리(ld)가 이보다 짧으면 curvature = 2*sin(alpha)/ld 계산의
        # 분모(ld)를 이 값으로 바닥을 깐다 — 유효 슬라이스가 적어 path가 짧게 잘린 프레임
        # (부분 가림, 순간 노이즈 등)에서 _target_point()가 (가변)lookahead에 못 미쳐
        # path[-1](아주 가까운 점)을 목표점으로 쓰게 되면, ld가 작을수록 같은 dx도 훨씬
        # 큰 곡률로 증폭된다. 실측 예: ld=42px, dx=3px(육안으론 거의 직진)여도 alpha≈4.1°,
        # curvature≈0.0034 → atan(curvature*wheelbase_px=220)≈37° — 픽셀 몇 개짜리 잡음이
        # 30도대 조향 스파이크로 증폭되는 걸 실제로 재현 가능.
        #   [2026-08-06] 예전엔 ld<min_lookahead_px일 때 계산 자체를 건너뛰고 직전 조향각을
        #   그대로 반환했는데(freeze), 이러면 "ld가 짧은 상태"(주로 dx 자체가 작을 때 걸림,
        #   ld≥|dx|라서)가 진동 중 우연히 한 번 걸리는 순간 그때의 조향각에 영원히 고정돼
        #   버렸다 — 그 뒤로는 실제 편차가 계속 커져도 재계산을 아예 안 해서 차가 한쪽으로
        #   계속 밀리는데도 못 돌아오는 문제가 실차에서 재현됨(진동하다 갑자기 한쪽으로
        #   빠져서 그대로 직진하는 증상). 분모만 바닥을 깔면 노이즈 증폭은 여전히 막으면서도
        #   매 프레임 계속 살아있는 보정을 하므로 이 "얼어붙음"이 없어진다. 대신 짧은
        #   ld 구간에서 완전히 무시되던 미세한 잔떨림이 약하게나마 계속 반영될 수 있고,
        #   목표점이 아주 가까우면서 옆으로도 벌어져 있는 극단적인 경우엔(alpha 자체가
        #   크면) 여전히 큰 조향각이 나올 수 있다 — alpha는 이 바닥값의 영향을 안 받기
        #   때문. 실차 미검증 튜닝값, 저속·개입 가능 상태로 재검증할 것.
        self.min_lookahead_px = min_lookahead_px

        # 프레임 간 조향각 스파이크 저역통과 — controller/stanley.py의 self.alpha와
        # 동일한 패턴(1프레임짜리 경로 튐이 조향에 그대로 실리지 않도록).
        self.alpha = alpha
        self.prev_steer_deg = 0.0
        # 직전 control() 호출이 "새로 계산"했는지 "직전값을 그대로 유지"했는지 표시.
        # track_drive.py의 DEBUG_VIZ_STEER 디버그 창이 이 값을 읽어서 보여준다.
        self.held = False
        # 직전에 새로 계산됐을 때의 curvature(=2*sin(alpha)/ld, 1/px 단위) — 조향각으로
        # 변환되고 나면 사라지는 값이라 별도로 보관해둔다. track_drive.py가 코너 진입 시
        # 감속량을 정하는 데 쓴다(ROS2 Nav2의 Regulated Pure Pursuit과 같은 발상: 회전반경이
        # 작아질수록 속도를 줄여서, 짧은 lookahead에서 픽셀 노이즈가 조향으로 증폭되는 걸
        # "속도를 늦춰 반응시간을 버는" 방식으로도 완화한다). held=True인 프레임에는
        # 갱신하지 않고 직전값을 그대로 유지한다 — prev_steer_deg와 같은 원칙.
        self.last_curvature = 0.0

    def reset(self):
        self.prev_steer_deg = 0.0
        self.held = False
        self.last_curvature = 0.0

    def _target_point(self, path, vehicle_xy, lookahead_px):
        """path를 따라 vehicle_xy로부터 누적 호길이가 lookahead_px를 넘는 첫 지점을
        찾아 그 직전 구간 위에서 선형보간한다. 경로 전체 길이가 lookahead_px보다
        짧으면(짧은 ROI, 혹은 유효 슬라이스가 적어 path가 짧게 잘린 경우) 경로의
        가장 먼 점을 그대로 목표점으로 쓴다."""
        acc = 0.0
        prev = vehicle_xy
        for x, y in path:
            seg = math.hypot(x - prev[0], y - prev[1])
            if seg > 1e-6 and acc + seg >= lookahead_px:
                t = (lookahead_px - acc) / seg
                return (prev[0] + t * (x - prev[0]), prev[1] + t * (y - prev[1]))
            acc += seg
            prev = (x, y)
        return path[-1]

    def control(self, path, vehicle_xy, speed=0.0):
        """path : [(x,y), ...] ROI 픽셀좌표, 가까운점(차량 근처)→먼점 순
           vehicle_xy : (x,y) 차량 기준점(관례상 ROI 하단 중앙 == path[0] 근방)
           speed : 직전 프레임 명령속도(track_drive.py의 _prev_speed, 모터 단위) 근사치 —
                   lookahead를 속도에 맞춰 늘이고 줄이는 데만 쓰는 순간값이라 누적하지
                   않는다(위치를 적분하는 dead-reckoning과는 다름, 드리프트 없음).
           반환 : 조향각(도), ±angle_max_deg로 클램프.
           호출 후 self.held로 이번 호출이 "새로 계산"(False)했는지 "직전값 유지"(True)
           했는지, self.last_curvature로 이번(혹은 마지막 유효) curvature를 확인할 수
           있다(디버그 창/코너 감속용)."""
        if not path:
            self.held = True
            return self.prev_steer_deg

        lookahead_px = float(np.clip(
            self.lookahead_base_px + self.lookahead_speed_gain * max(speed, 0.0),
            self.lookahead_base_px, self.lookahead_max_px
        ))
        tx, ty = self._target_point(path, vehicle_xy, lookahead_px)
        dx = tx - vehicle_xy[0]
        if abs(dx) < self.dx_deadzone_px:
            dx = 0.0
        # 이미지 y는 아래로 증가하므로 "전방으로 이만큼"은 vehicle_xy[1]-ty(위로 갈수록 +).
        # 목표점이 차량과 같은 행(dy<=0)에 있는 뒤틀린 경로라도 나눗셈이 죽지 않도록
        # 최소값을 준다.
        dy = max(vehicle_xy[1] - ty, 1e-3)
        # ld가 너무 짧으면(위 min_lookahead_px 주석 참고) 분모만 바닥을 깐다 — 계산 자체를
        # 건너뛰지 않는다. 짧은 ld에서 픽셀 노이즈가 조향각으로 증폭되는 건 막으면서도,
        # 매 프레임 계속 보정하므로 "얼어붙어서 못 돌아오는" 문제가 없다.
        ld = max(math.hypot(dx, dy), self.min_lookahead_px)

        alpha = math.atan2(dx, dy)
        curvature = 2.0 * math.sin(alpha) / ld
        steer_deg = math.degrees(math.atan(curvature * self.wheelbase_px))

        steer_deg = self.alpha * steer_deg + (1.0 - self.alpha) * self.prev_steer_deg
        steer_deg = float(np.clip(steer_deg, -self.angle_max_deg, self.angle_max_deg))
        self.prev_steer_deg = steer_deg
        self.held = False
        self.last_curvature = curvature
        return steer_deg
