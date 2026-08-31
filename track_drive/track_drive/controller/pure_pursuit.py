#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#=============================================
# pure_pursuit.py — DL/classic 세그멘테이션이 만든 BEV/ROI 픽셀 경로를 따라가는
# 기하학적(geometric) 조향 컨트롤러.
#
# lane_util._fit_and_sample_path()가 슬라이스별 차선 중심점에 다항식을 피팅해 만든
# 명시적 곡선(웨이포인트 리스트, ROI 픽셀좌표, 가까운점→먼점 순)을 그대로 받아
# Pure Pursuit 기하 공식으로 조향각을 계산한다.
#
# ── 좌표계: 왜 미터가 아니라 픽셀인가 ──
#   controller/stanley.py는 planner/hybrid_astar.py가 만든 실세계 미터 단위 경로를
#   차량 위치(x,y,yaw)와 함께 추종한다. 여기서는 그게 불가능한데, 카메라 픽셀→미터
#   변환이 아직 실측 전(0.0, classic_cv 백엔드 기준)이기 때문이다. 그래서 차량은 항상
#   ROI 하단 중앙(vehicle_xy, path[0] 근방)에 고정된 것으로 보고, 카메라가 보는 위쪽
#   방향(-y)을 "전방"으로 삼아 픽셀 좌표계 안에서 그대로 기하 계산을 한다.
#   단, 기본 조합(LANE_DETECTOR_BACKEND='dl' + DL_USE_BEV=True)에서는 self.lane_path가
#   정확히 config.DL_PIXELS_PER_METER(=200px/m) 스케일이므로, wheelbase_px는 실측
#   WHEELBASE_M(0.335m)*DL_PIXELS_PER_METER로 계산한 물리 기반 값(67.0)을 쓴다
#   (config.py PP_WHEELBASE_PX 주석 참고) — 여전히 픽셀 좌표계에서 계산하지만 게인
#   자체는 실측에 근거한다. 실차 재검증 필요.
#=============================================
import math

import numpy as np


class PurePursuitController:
    """path(ROI 픽셀좌표 웨이포인트, 가까운점→먼점)와 차량 기준점을 받아 조향각(도)을
    반환한다. path가 비어있으면(첫 프레임 등) 직전 조향각을 그대로 유지한다 —
    lane_util._debounce()가 무효 프레임에 직전 확정값을 유지하는 것과 같은 원칙."""

    def __init__(self, lookahead_base_px=90.0, lookahead_speed_gain=4.0, lookahead_max_px=150.0,
                 wheelbase_px=67.0, angle_max_deg=100.0, alpha=0.5,
                 ld_floor_px=90.0, dx_deadzone_px=6.0, lookahead_curvature_gain=100.0,
                 lookahead_min_px=40.0,
                 wheelbase_boost_enable=False,
                 wheelbase_boost_gain_per_deg=0.02, wheelbase_boost_max_scale=1.5,
                 lookahead_alpha=1.0, lookahead_speed_anchor=0.0):
        # 속도 적응형 lookahead. curvature ≈ 2*dx/ld^2(작은각 근사)라서 ld가 짧을수록
        # 같은 dx도 1/ld^2로 증폭된다 — 비전 기반 경로는 매 프레임 픽셀 노이즈가 있어
        # 짧은 lookahead를 쓰면 그 노이즈까지 증폭된다(실차에서 waypoint가 살짝만
        # 틀어져도 조향각이 크게 튀는 증상으로 확인됨). base(90px)를 하한으로 두고
        # 속도가 오르면 그 위로만 늘어나게 한다(절대 base 밑으로는 안 내려감). 저속에서
        # tight-tracking 이점은 없지만 노이즈 증폭이 없는 쪽을 우선한 것 — 실차에서 저속
        # 코너 추종이 둔하면 base보다 wheelbase_px를 낮추는 쪽이 더 직접적인 레버다
        # (하단 주석 참고). gain=4.0은 SPEED_NORMAL(5.0)에서 90+4*5=110, 최고속도가
        # 올라도 max_px(150) 여유 안에 들어오도록 잡은 값. 전부 실차 미검증 튜닝값.
        self.lookahead_base_px = lookahead_base_px
        self.lookahead_speed_gain = lookahead_speed_gain
        self.lookahead_max_px = lookahead_max_px
        # speed_lookahead_px = base + gain*(speed - anchor)의 anchor(config.py
        # PP_LOOKAHEAD_SPEED_ANCHOR 참고) — 기본값 0.0이면 base가 "speed=0 기준".
        # anchor를 SPEED_NORMAL로 주면 base 자체가 "주행속도 기준" lookahead값이 되어
        # config.py에서 더 직관적으로 읽힌다. speed15 프리셋만 anchor=SPEED_NORMAL(12.0)
        # 로 설정, 다른 프리셋/기본값은 anchor=0.0(하위호환).
        self.lookahead_speed_anchor = lookahead_speed_anchor

        # curvature 기반 lookahead 감쇠 — 직전 프레임이 타이트한 코너였으면(curvature 큼)
        # 이번 프레임 lookahead를 속도가 밀어올린 값보다 낮춰서(lookahead_min_px 쪽으로
        # 당겨서) 더 촘촘하게 추종하고, 직진이면(curvature≈0) 원래 속도 기반 값을 그대로
        # 쓴다. gain=100.0은 curvature=0.01(반경≈100px, 중간 코너)에서 배율이 0.5로
        # 떨어지도록 잡은 추정치 — 실차 미검증, steer_debug로 관찰하며 조정할 것.
        # 주의: 이 damp를 그대로 self.last_curvature 재사용으로 계산하면 자기순환
        # 피드백에 빠진다 — ld가 한번 짧아지면 같은 픽셀 노이즈도 curvature 공식에서
        # 더 크게 증폭되고, 그 커진 curvature가 다음 프레임 lookahead를 또 줄여 실제
        # 경로가 직진으로 돌아와도 원래 lookahead로 복귀를 못 하는 락업이 생긴다(실차
        # 디버그 영상에서 조향각이 15초 넘게 고정되는 것으로 재현됨). control()은 이
        # 순환을 끊기 위해 damp 판단용 probe_curvature를 매 프레임 "댐핑 적용 전 고정
        # 기준 lookahead"로 별도 계산한다 — 아래 control() 내부 주석 참고.
        self.lookahead_curvature_gain = lookahead_curvature_gain
        # 위 감쇠가 당길 수 있는 하한 — ld_floor_px(curvature 계산의 ld 분모 바닥값,
        # 완전히 다른 용도)와 절대 헷갈리지 말 것. 둘 다 90.0이면 damp가 아무리 작아져도
        # lookahead_px가 base(90)로 다시 clip돼서 이 기능 자체가 무효화된다 — 그래서
        # 반드시 lookahead_base_px보다 뚜렷이 작은 값으로 따로 둔다. 40px은 추정치,
        # 실차에서 코너 추종이 여전히 둔하면 더 낮출 것.
        self.lookahead_min_px = lookahead_min_px

        # lookahead_px 자체에 거는 프레임간 저역통과 — self.alpha가 이미 steer_deg에
        # 필터를 걸고 있지만, lookahead_px는 그 필터를 타기 "전" 단계라 curvature_damp가
        # 한 프레임 만에 base에서 min_px로 뚝 떨어질 수 있고, 그 좁아진 lookahead가 같은
        # dx도 더 큰 curvature로 증폭시켜(curvature=2*sin(alpha)/ld) 스파이크를 오히려
        # 키우거나 복귀 중 재흔들림을 만들 수 있다(위 lookahead_curvature_gain 주석 참고).
        # 1.0(기본값)이면 필터 없음. 1보다 작으면 lookahead_px가 여러 프레임에 걸쳐
        # 서서히 줄었다 늘었다 한다. control() 하단에서 적용, prev는
        # self.last_lookahead_px(다른 last_* 필드와 동일하게 held=True 프레임엔 유지).
        # 실차 미검증.
        self.lookahead_alpha = lookahead_alpha

        # dx(waypoint-차량중앙 픽셀오차)가 이 값 미만이면 0으로 죽인다 — 차량이 차선
        # 중앙에 사실상 있는데도(세그멘테이션의 서브픽셀 흔들림) 매 프레임 미세하게 다른
        # 조향이 나가 중앙 부근에서 계속 잔떨림하는 것을 막는다. _lane_pid()의
        # LANE_DEADZONE(40px, PID 입력용)과 같은 철학이지만 여긴 훨씬 작은 값을 쓴다 —
        # Pure Pursuit은 목표점 자체가 이미 lookahead 앞의 실제 경로점이라 LANE_DEADZONE
        # 만큼 크게 죽이면 완만한 커브 진입까지 무시하게 된다. 실차 미검증.
        self.dx_deadzone_px = dx_deadzone_px

        # "곡률→조향각" 게인. 표준 Pure Pursuit 공식 steer = atan(2*L*sin(alpha)/Ld)에서
        # L 자리에 들어간다 — 크게 잡을수록 같은 곡률에도 조향각이 커진다(더 공격적).
        # 기본 조합(dl+BEV)에서는 self.lane_path가 DL_PIXELS_PER_METER(200px/m) 스케일
        # 이라, 실측 WHEELBASE_M(0.335m)*200=67.0의 물리 기반 값을 쓴다(config.py
        # PP_WHEELBASE_PX 참고) — 그래도 실차에서 직진 안정성/코너 추종성을 보고
        # 미세조정할 것. lookahead보다 이 값을 낮추는 쪽이 조향 민감도를 낮추는 더
        # 직접적인 레버다 — curvature*wheelbase_px에 그대로 곱해지므로 ld 범위 전체에서
        # 균일하게 작용한다(단, 실제 코너에서도 반응이 약해지니 함께 재확인할 것).
        self.wheelbase_px = wheelbase_px

        # 조향각이 클수록 wheelbase_px를 비례해서 키우는 "부스트"(config.py
        # PP_WHEELBASE_BOOST_* 참고) — 1차 steer_deg의 절댓값에 비례해서(도당
        # gain_per_deg 비율) wheelbase_px를 키워 atan 포화 쪽으로 더 밀어붙인다
        # (max_scale로 상한). 각도 0에서부터 연속적으로(선형) 커지므로 각도 0이면
        # 배율도 정확히 1.0(무보정)이라 계단형 문턱 같은 "임계 넘는 순간 갑자기 세짐"
        # 현상이 없다. wheelbase_boost_enable은 "speed15" 프리셋(PP_TUNE_ACTIVE_PRESET)
        # 일 때만 track_drive.py가 True로 넘긴다 — 순간 명령/실측 속도값이 아니라 어떤
        # 튜닝 프리셋을 쓰고 있는지로 켜고 끈다. 실차 미검증.
        self.wheelbase_boost_enable = wheelbase_boost_enable
        self.wheelbase_boost_gain_per_deg = wheelbase_boost_gain_per_deg
        self.wheelbase_boost_max_scale = wheelbase_boost_max_scale

        self.angle_max_deg = angle_max_deg

        # 목표점까지 남은 실제거리(ld)가 이보다 짧으면 curvature = 2*sin(alpha)/ld 계산의
        # 분모(ld)를 이 값으로 바닥을 깐다 — 유효 슬라이스가 적어 path가 짧게 잘린 프레임
        # (부분 가림, 순간 노이즈 등)에서 _target_point()가 (가변)lookahead에 못 미쳐
        # path[-1](아주 가까운 점)을 목표점으로 쓰게 되면, ld가 작을수록 같은 dx도 훨씬
        # 큰 곡률로 증폭된다. 실측 예: ld=42px, dx=3px(육안으론 거의 직진)여도 alpha≈4.1°,
        # curvature≈0.0034 → atan(curvature*wheelbase_px=220)≈37° — 픽셀 몇 개짜리 잡음이
        # 30도대 조향 스파이크로 증폭되는 걸 실제로 재현 가능.
        # 분모만 바닥을 까는 이유: ld<ld_floor_px일 때 계산 자체를 건너뛰고 직전 조향각을
        # 그대로 반환(freeze)하면, 진동 중 우연히 한 번 걸리는 순간의 조향각에 영원히
        # 고정돼버려 그 뒤로 편차가 계속 커져도 차가 못 돌아오는 문제가 실차에서
        # 재현됐다. 분모만 바닥을 깔면 노이즈 증폭은 막으면서도 매 프레임 계속 보정이
        # 살아있어 이 "얼어붙음"이 없다. 대신 목표점이 아주 가까우면서 옆으로도 크게
        # 벌어진 극단적인 경우엔(alpha 자체가 크면) 여전히 큰 조향각이 나올 수 있다 —
        # alpha는 이 바닥값의 영향을 안 받기 때문. 실차 미검증 튜닝값, 저속·개입 가능
        # 상태로 재검증할 것.
        self.ld_floor_px = ld_floor_px

        # 프레임 간 조향각 스파이크 저역통과 — controller/stanley.py의 self.alpha와
        # 동일한 패턴(1프레임짜리 경로 튐이 조향에 그대로 실리지 않도록).
        self.alpha = alpha
        self.prev_steer_deg = 0.0
        # 직전 control() 호출이 "새로 계산"했는지 "직전값을 그대로 유지"했는지 표시.
        # track_drive.py의 DEBUG_VIZ_STEER 디버그 창이 이 값을 읽어서 보여준다.
        self.held = False
        # 직전에 새로 계산된 lookahead 목표점(ROI/da 픽셀좌표)과 그때 쓴 lookahead_px —
        # track_drive.py가 dl_lane.py의 'result' 패널에 그대로 찍어서(속도 적응형 lookahead가
        # 실제로 어디를 보고 있는지) 디버깅할 수 있게 노출한다. path가 비어 held=True인
        # 프레임에는 갱신하지 않고 직전값을 그대로 들고 있는다(다른 last_* 필드와 동일 원칙).
        self.last_target_xy = None
        self.last_lookahead_px = None
        # 직전에 새로 계산됐을 때의 curvature(=2*sin(alpha)/ld, 1/px 단위) — 조향각으로
        # 변환되고 나면 사라지는 값이라 별도로 보관해둔다. track_drive.py가 코너 진입 시
        # 감속량을 정하는 데 쓴다(ROS2 Nav2의 Regulated Pure Pursuit과 같은 발상: 회전반경이
        # 작아질수록 속도를 줄여서, 짧은 lookahead에서 픽셀 노이즈가 조향으로 증폭되는 걸
        # "속도를 늦춰 반응시간을 버는" 방식으로도 완화한다). held=True인 프레임에는
        # 갱신하지 않고 직전값을 그대로 유지한다 — prev_steer_deg와 같은 원칙.
        self.last_curvature = 0.0
        # 이번(혹은 마지막 유효) 프레임에서 실제로 damp 판단에 쓰인 IMU curvature(1/px) —
        # track_drive.py._imu_curvature_px()가 넘겨준 값(IMU/VESC 둘 다 살아있고 dl+BEV
        # 조합일 때만 not None). 디버그 창에서 "지금 IMU가 감쇠에 실제로 기여하는지"를
        # 바로 확인할 수 있게 노출한다 — control()에 매번 넘어와도 probe_curvature가 더
        # 크면 반영이 안 될 수 있으니 last_curvature만으로는 구분이 안 됨.
        self.last_imu_curvature_px = None
        # wheelbase 부스트(위 wheelbase_boost_* 주석 참고) 적용 "전" 1차
        # steer_deg — track_drive.py가 dl_lane.py 'll' 패널에 "보정 전/후" 조향각을 같이
        # 찍어서 부스트가 실제로 얼마나 개입 중인지 한눈에 보여주는 데 쓴다. prev_steer_deg는
        # 부스트+필터+클램프까지 다 거친 "실제로 나가는" 값이라, 부스트 자체의 기여도만
        # 따로 보려면 이 값과 같이 봐야 한다. held=True인 프레임에는 갱신하지 않는다(다른
        # last_* 필드와 동일 원칙).
        self.last_pre_boost_steer_deg = 0.0

    def reset(self):
        self.prev_steer_deg = 0.0
        self.held = False
        self.last_curvature = 0.0
        self.last_target_xy = None
        self.last_lookahead_px = None
        self.last_imu_curvature_px = None
        self.last_pre_boost_steer_deg = 0.0

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

    def _target_point_max_deviation(self, path, vehicle_xy, lookahead_px):
        """근접 장애물 급회피 대응 — _target_point()가 "호길이 lookahead_px 지점 딱
        한 점"만 보는 것과 달리, 차량~lookahead_px 구간 전체를 훑으면서 가로편차
        (|x-vehicle_x|)가 가장 큰 점을 목표점으로 삼는다.

        왜 필요한가: da 안전마진을 넓힌 조합에서는 경로가 근접부에서 급하게
        옆차선 쪽으로 꺾이고 그 뒤로는 완만해지는(비단조적) 모양이 나올 수 있다.
        기존 _target_point()는 그 급한 꺾임을 지나쳐 먼 지점의 "각도"만 보므로
        (alpha가 커도 ld가 커서 curvature=2*sin(alpha)/ld가 작게 나옴), 실제로는
        지금 당장 크게 틀어야 하는데도 조향이 완만하게만 걸려 충돌하는 문제가
        실차에서 재현됨.

        단조롭게 휘는 보통 커브(장애물 유무 무관)에서는 편차가 거리에 비례해
        커지므로 최대편차점이 결국 _target_point()가 고르는 가장 먼 점과 같아져
        결과가 사실상 동일하다 — 이 메서드가 실제로 다른 결과를 내는 건 경로가
        진짜로 비단조적일 때뿐이다. 그래서 control()은 이 메서드를 상시 쓰지
        않고 근접 장애물이 있을 때만 한정적으로 쓴다(호출부 주석 참고) — 평상시
        주행에 쓰던 검증된 _target_point()의 안정성을 건드리지 않기 위함.

        _target_point()와 동일한 경계 처리(호길이 lookahead_px 지점에서 선형보간,
        경로가 짧으면 마지막 점)를 그대로 따르되, 그 경계점까지 지나온 모든 점과
        경계점 자신 중 편차 최댓값을 반환한다. 실차 미검증."""
        acc = 0.0
        prev = vehicle_xy
        best_pt = path[-1] if path else vehicle_xy
        best_dev = -1.0
        for x, y in path:
            seg = math.hypot(x - prev[0], y - prev[1])
            if seg > 1e-6 and acc + seg >= lookahead_px:
                t = (lookahead_px - acc) / seg
                boundary = (prev[0] + t * (x - prev[0]), prev[1] + t * (y - prev[1]))
                dev = abs(boundary[0] - vehicle_xy[0])
                if dev > best_dev:
                    best_pt = boundary
                return best_pt
            dev = abs(x - vehicle_xy[0])
            if dev > best_dev:
                best_dev = dev
                best_pt = (x, y)
            acc += seg
            prev = (x, y)
        return best_pt

    def control(self, path, vehicle_xy, speed=0.0, imu_curvature_px=None, near_obstacle=False):
        """path : [(x,y), ...] ROI 픽셀좌표, 가까운점(차량 근처)→먼점 순
           vehicle_xy : (x,y) 차량 기준점(관례상 ROI 하단 중앙 == path[0] 근방)
           speed : 직전 프레임 명령속도(track_drive.py의 _prev_speed, 모터 단위) 근사치 —
                   lookahead를 속도에 맞춰 늘이고 줄이는 데만 쓰는 순간값이라 누적하지
                   않는다(위치를 적분하는 dead-reckoning과는 다름, 드리프트 없음).
           imu_curvature_px : IMU 각속도(yaw_rate) + VESC 실측속도로 구한 "차량이 지금
                   실제로 얼마나 도는지" curvature(1/px, 부호 불문 — abs로만 쓰임).
                   track_drive.py._imu_curvature_px()가 IMU/VESC가 둘 다 살아있고
                   dl+BEV 조합일 때만 값을 주고, 그 외(둘 중 하나라도 죽어있거나 다른
                   백엔드)엔 None을 줘서 아래 로직이 기존처럼 probe_curvature 단독
                   판단으로 자동 폴백하게 만든다.
           near_obstacle : True면 목표점 선택을 _target_point() 대신
                   _target_point_max_deviation()으로 바꾼다(그 메서드 docstring 참고) —
                   기본값 False라 이 인자를 안 넘기는 기존 호출부(track_drive.py
                   _handle_lavacon() 등)는 전과 완전히 동일하게 동작한다. 그 외
                   lookahead/댐핑/필터/클램프 로직은 이 값과 무관하게 전부 동일하다.
           반환 : 조향각(도), ±angle_max_deg로 클램프.
           호출 후 self.held로 이번 호출이 "새로 계산"(False)했는지 "직전값 유지"(True)
           했는지, self.last_curvature로 이번(혹은 마지막 유효) curvature를 확인할 수
           있다(디버그 창/코너 감속용)."""
        if not path:
            self.held = True
            return self.prev_steer_deg

        speed_lookahead_px = self.lookahead_base_px + self.lookahead_speed_gain * (
            max(speed, 0.0) - self.lookahead_speed_anchor)

        # damp 판단은 "직전에 실제로 쓴(이미 줄어들었을 수 있는) lookahead에서 나온
        # self.last_curvature"가 아니라, 매 프레임 댐핑 적용 전 고정 기준(probe_px)에서
        # 새로 계산한 probe_curvature로 한다 — self.last_curvature를 재사용하면 자기순환
        # 피드백 락업이 생긴다(클래스 상단 lookahead_curvature_gain 주석 참고). probe는
        # 항상 같은 기준거리에서 다시 재므로 "지금 경로가 진짜 코너인지"를 매 프레임
        # 독립적으로 재평가한다 — 직진이면(probe_curvature≈0) damp≈1이라 기존과 동일하게
        # 동작한다.
        probe_px = float(np.clip(speed_lookahead_px, self.lookahead_min_px, self.lookahead_max_px))
        probe_tx, probe_ty = self._target_point(path, vehicle_xy, probe_px)
        probe_dx = probe_tx - vehicle_xy[0]
        probe_dy = max(vehicle_xy[1] - probe_ty, 1e-3)
        probe_ld = max(math.hypot(probe_dx, probe_dy), min(self.ld_floor_px, probe_px))
        probe_curvature = 2.0 * math.sin(math.atan2(probe_dx, probe_dy)) / probe_ld
        # IMU 실측 curvature 반영 — probe_curvature는 "아직 안 가본 앞쪽 경로가 얼마나
        # 휘었는지"에 대한 비전 추정치라, 경로 자체가 흔들리면 같이 흔들린다.
        # imu_curvature_px는 "차량이 지금 실제로 얼마나 돌고 있는지" 실측값이라 이
        # 노이즈가 없다. 둘 중 하나만 믿기보다 둘 중 절댓값이 더 큰 쪽으로 damp를 건다 —
        # 비전이 못 본 코너를 IMU가 잡아내는 경우(혹은 그 반대)를 놓치지 않기 위한
        # 보수적(더 세게 죽이는 쪽) 선택. imu_curvature_px가 None이면(IMU/VESC 중
        # 하나라도 죽어있거나 dl+BEV 조합이 아니면) 기존과 동일하게 probe_curvature
        # 단독으로 판단한다.
        damp_curvature = abs(probe_curvature)
        if imu_curvature_px is not None:
            damp_curvature = max(damp_curvature, abs(imu_curvature_px))
        self.last_imu_curvature_px = imu_curvature_px
        curvature_damp = 1.0 / (1.0 + self.lookahead_curvature_gain * damp_curvature)

        lookahead_px_raw = float(np.clip(
            speed_lookahead_px * curvature_damp,
            self.lookahead_min_px, self.lookahead_max_px
        ))
        # lookahead_px 저역통과(__init__ lookahead_alpha 주석 참고) — 직전 프레임에 새로
        # 계산됐던 lookahead(self.last_lookahead_px)와 블렌딩한다. held=True로 건너뛴
        # 프레임 직후에는 last_lookahead_px가 "몇 프레임 전" 값일 수 있지만, 그것도
        # "직전에 유효했던 값"이라는 점에서 prev_steer_deg와 같은 원칙이라 문제 없다.
        if self.last_lookahead_px is None:
            lookahead_px = lookahead_px_raw
        else:
            lookahead_px = (self.lookahead_alpha * lookahead_px_raw
                             + (1.0 - self.lookahead_alpha) * self.last_lookahead_px)
        if near_obstacle:
            tx, ty = self._target_point_max_deviation(path, vehicle_xy, lookahead_px)
        else:
            tx, ty = self._target_point(path, vehicle_xy, lookahead_px)
        self.last_target_xy = (tx, ty)
        self.last_lookahead_px = lookahead_px
        dx = tx - vehicle_xy[0]

        if abs(dx) < self.dx_deadzone_px:
            dx = 0.0
        # 이미지 y는 아래로 증가하므로 "전방으로 이만큼"은 vehicle_xy[1]-ty(위로 갈수록 +).
        # 목표점이 차량과 같은 행(dy<=0)에 있는 뒤틀린 경로라도 나눗셈이 죽지 않도록
        # 최소값을 준다.
        dy = max(vehicle_xy[1] - ty, 1e-3)
        # ld가 너무 짧으면(위 ld_floor_px 주석 참고) 분모만 바닥을 깐다 — 계산 자체를
        # 건너뛰지 않는다. 짧은 ld에서 픽셀 노이즈가 조향각으로 증폭되는 건 막으면서도,
        # 매 프레임 계속 보정하므로 "얼어붙어서 못 돌아오는" 문제가 없다.
        # 바닥값을 ld_floor_px로 고정하지 않는 이유: curvature 기반 감쇠(위
        # lookahead_curvature_gain 주석 참고)로 이번 프레임 lookahead_px가 ld_floor_px
        # (90)보다 작게(코너에서 의도적으로) 정해졌는데 바닥을 그대로 90으로 두면, ld가
        # 거의 항상 그 90으로 다시 눌려서 "코너에서 더 촘촘히 추종한다"는 효과가
        # 없어진다. 그래서 바닥을 min(ld_floor_px, lookahead_px)로 잡는다 — 직진처럼
        # lookahead_px가 ld_floor_px 이상이면 기존과 동일하게 90이 바닥이고, 코너처럼
        # lookahead_px를 일부러 줄였으면 그 줄어든 값 자체가 바닥이 된다.
        ld = max(math.hypot(dx, dy), min(self.ld_floor_px, lookahead_px))

        alpha = math.atan2(dx, dy)
        curvature = 2.0 * math.sin(alpha) / ld
        steer_deg = math.degrees(math.atan(curvature * self.wheelbase_px))
        self.last_pre_boost_steer_deg = steer_deg

        # wheelbase 부스트(위 __init__ 주석 참고) — "speed15" 프리셋일 때만
        # (wheelbase_boost_enable) 1차 steer_deg의 절댓값에 비례해서(문턱 없음, 각도 0이면
        # boost_scale도 정확히 1.0=무보정) wheelbase_px를 키워 같은 curvature로 다시
        # 계산한다. 아래 필터(self.alpha)/클램프는 이 boosted steer_deg에 그대로 이어서
        # 적용된다 — 기존 흐름과 동일.
        if self.wheelbase_boost_enable:
            boost_scale = min(self.wheelbase_boost_max_scale,
                               1.0 + self.wheelbase_boost_gain_per_deg * abs(steer_deg))
            steer_deg = math.degrees(math.atan(curvature * self.wheelbase_px * boost_scale))

        steer_deg = self.alpha * steer_deg + (1.0 - self.alpha) * self.prev_steer_deg
        steer_deg = float(np.clip(steer_deg, -self.angle_max_deg, self.angle_max_deg))
        self.prev_steer_deg = steer_deg
        self.held = False
        self.last_curvature = curvature
        return steer_deg
