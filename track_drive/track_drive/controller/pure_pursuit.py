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

    def __init__(self, lookahead_px=90.0, wheelbase_px=50.0, angle_max_deg=100.0, alpha=0.5,
                 min_lookahead_px=65.0):
        # 목표점을 찾는 전방주시거리(px). ROI가 짧은 백엔드(hough_lane은 ROI 높이가
        # ~70px로 매우 짧다)에서는 경로 전체 길이가 lookahead_px보다 짧아져 자동으로
        # path의 가장 먼 점이 목표점이 된다(_target_point() 참고) — 그 자체로는 안전한
        # 폴백이지만 반응이 둔해지니, 그런 백엔드에서 체감 반응이 느리면 값을 줄일 것.
        # 실차 미검증 튜닝값.
        self.lookahead_px = lookahead_px

        # 실제 차축거리(m) 대신 쓰는 "곡률→조향각" 게인. 표준 Pure Pursuit 공식
        # steer = atan(2*L*sin(alpha)/Ld)에서 L 자리에 들어간다 — 크게 잡을수록 같은
        # 곡률에도 조향각이 커진다(더 공격적). PIXELS_PER_METER 실측 전까지는 물리적
        # 의미가 없는 순수 튜닝 게인이므로, 실차에서 직진 안정성/코너 추종성을 보고
        # 맞출 것. 실차 미검증 튜닝값.
        self.wheelbase_px = wheelbase_px

        self.angle_max_deg = angle_max_deg

        # 목표점까지 남은 실제거리(ld)가 이보다 짧으면 곡률 계산을 하지 않고 직전 조향각을
        # 유지한다 — 유효 슬라이스가 적어 path가 짧게 잘린 프레임(부분 가림, 순간 노이즈
        # 등)에서 _target_point()가 lookahead_px에 못 미쳐 path[-1](아주 가까운 점)을
        # 목표점으로 쓰게 되는데, curvature = 2*sin(alpha)/ld 는 ld가 작을수록 같은 dx도
        # 훨씬 큰 곡률로 증폭시킨다. 실측 예: ld=42px, dx=3px(육안으론 거의 직진)여도
        # alpha≈4.1°, curvature≈0.0034 → atan(curvature*wheelbase_px=220)≈37° — 픽셀
        # 몇 개짜리 잡음이 30도대 조향 스파이크로 증폭되는 걸 실제로 재현 가능. path가
        # 비었을 때(위 return self.prev_steer_deg)와 같은 "못 믿을 프레임은 직전 값 유지"
        # 원칙을 짧은 ld에도 동일하게 적용한다. 실차 미검증 튜닝값.
        self.min_lookahead_px = min_lookahead_px

        # 프레임 간 조향각 스파이크 저역통과 — controller/stanley.py의 self.alpha와
        # 동일한 패턴(1프레임짜리 경로 튐이 조향에 그대로 실리지 않도록).
        self.alpha = alpha
        self.prev_steer_deg = 0.0

    def reset(self):
        self.prev_steer_deg = 0.0

    def _target_point(self, path, vehicle_xy):
        """path를 따라 vehicle_xy로부터 누적 호길이가 lookahead_px를 넘는 첫 지점을
        찾아 그 직전 구간 위에서 선형보간한다. 경로 전체 길이가 lookahead_px보다
        짧으면(짧은 ROI, 혹은 유효 슬라이스가 적어 path가 짧게 잘린 경우) 경로의
        가장 먼 점을 그대로 목표점으로 쓴다."""
        acc = 0.0
        prev = vehicle_xy
        for x, y in path:
            seg = math.hypot(x - prev[0], y - prev[1])
            if seg > 1e-6 and acc + seg >= self.lookahead_px:
                t = (self.lookahead_px - acc) / seg
                return (prev[0] + t * (x - prev[0]), prev[1] + t * (y - prev[1]))
            acc += seg
            prev = (x, y)
        return path[-1]

    def control(self, path, vehicle_xy):
        """path : [(x,y), ...] ROI 픽셀좌표, 가까운점(차량 근처)→먼점 순
           vehicle_xy : (x,y) 차량 기준점(관례상 ROI 하단 중앙 == path[0] 근방)
           반환 : 조향각(도), ±angle_max_deg로 클램프."""
        if not path:
            return self.prev_steer_deg

        tx, ty = self._target_point(path, vehicle_xy)
        dx = tx - vehicle_xy[0]
        # 이미지 y는 아래로 증가하므로 "전방으로 이만큼"은 vehicle_xy[1]-ty(위로 갈수록 +).
        # 목표점이 차량과 같은 행(dy<=0)에 있는 뒤틀린 경로라도 나눗셈이 죽지 않도록
        # 최소값을 준다.
        dy = max(vehicle_xy[1] - ty, 1e-3)
        ld = math.hypot(dx, dy)

        # ld가 너무 짧으면(위 min_lookahead_px 주석 참고) 곡률 계산 자체를 건너뛰고
        # 직전 조향각을 유지한다 — 짧은 ld에서는 픽셀 노이즈가 조향각으로 크게 증폭된다.
        if ld < self.min_lookahead_px:
            return self.prev_steer_deg

        alpha = math.atan2(dx, dy)
        curvature = 2.0 * math.sin(alpha) / ld
        steer_deg = math.degrees(math.atan(curvature * self.wheelbase_px))

        steer_deg = self.alpha * steer_deg + (1.0 - self.alpha) * self.prev_steer_deg
        steer_deg = float(np.clip(steer_deg, -self.angle_max_deg, self.angle_max_deg))
        self.prev_steer_deg = steer_deg
        return steer_deg
