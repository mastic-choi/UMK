'''
전방 차량 통과(추월/회피) 컨트롤러.

[대회 규정 근거 — 2026 9회 경주진행방법 p.26~33]
  · 차선은 2개, 가운데 노란 점선(넘어도 됨), 바깥은 흰 실선.
    "1,2차선 구분 없음", "점선을 두 바퀴 사이에 놓고 주행하는 것도 가능".
  · 고정장애물(p.30) = '고장난 차량'. 즉 정지 차량이고 크기는 방해차량과 같다.
  · 방해차량(p.32~33):
      - 한 대가 느린 속도로 주행하며 1차선과 2차선을 오간다.
      - "방해차량이 주행하지 않는 차선 쪽으로 추월해야 함"
      - 반대쪽으로 추월을 시도하면 차선이탈로 간주된다.
      - 추월 중에는 실선 이탈이 허용되지만, 추월 후 최대한 빨리 차선 안으로 복귀해야 하고
        뒷차와 90cm 이상 벌어진 시점에도 밖에 있으면 차선이탈 판정.
      - 추돌은 가해/피해 모두 벌초 10초.

[설계]
  두 미션은 결국 같은 기동이다 — "타겟이 없는 쪽으로 지나간다".
  다른 점은 재평가 빈도뿐이다:
    · 정지 타겟: 한번 방향을 정하면 그대로 유효하다.
    · 주행 타겟: 차선을 오가므로 매 프레임 다시 보고, 내가 가려는 쪽으로 타겟이
      넘어오면 방향을 바꾸거나 물러서야 한다.
  그래서 한 클래스로 만들고 moving 플래그로 재평가 강도만 바꾼다.

  정적/동적 판별을 위한 속도추정은 하지 않는다 — 상위 Phase 가 이미
  FIXED_OBSTACLE / VEHICLE 로 구분해 주기 때문이다(순차 미션 설계).
'''
from enum import Enum


# 반대 차선으로 이동할 목표 횡편차(px). ★B-1(차선 폭 실측) 후 실제 차선 폭으로 대체
PASS_OFFSET = 100.0

# 타겟 횡중심이 이 값(m) 이내면 '정면'으로 보고 방향을 다른 근거로 정한다
CENTER_DEADZONE_M = 0.12

# 진입/이탈 히스테리시스 (Apollo lane_borrow: 진입/이탈 비대칭 카운터)
CLEAR_FRAMES_TO_RETURN = 6     # 타겟이 안 보이는 상태가 이만큼 연속되면 복귀 시작
SWITCH_FRAMES = 8              # 주행 타겟이 내 진행쪽으로 넘어온 상태가 이만큼 지속되면 방향 전환

# 횡이동 수렴 (★3단계에서 5차 다항식·거리 매개변수로 교체 예정)
LATERAL_ALPHA_OUT = 0.12
LATERAL_ALPHA_BACK = 0.16      # 복귀는 더 빠르게 — 90cm 규정 때문에 늑장 부리면 차선이탈
LATERAL_DONE_PX = 8.0

# 추돌 방지 종방향 간격(m). 이보다 가까우면 횡이동이 끝날 때까지 속도를 죽인다
MIN_GAP_M = 0.6


class PassPhase(Enum):
    IDLE = 0
    SHIFT = 1      # 옆 차선으로 이동 중
    ALONGSIDE = 2  # 타겟 옆을 지나는 중
    RETURN = 3     # 원래 차선으로 복귀 중


class TargetPassing:
    """정지 차량·주행 차량 공통 통과 컨트롤러.

    update() → (steer_offset, speed, done, status)
      status: 'idle' / 'passing' / 'blocked'
    """

    def __init__(self, moving=False, preview=0.4,
                 shift_speed=8.0, alongside_speed=12.0, return_speed=10.0):
        self.moving = moving              # True = 방해차량(주행), False = 고장난 차량(정지)
        self.preview = preview
        self.shift_speed = shift_speed
        self.alongside_speed = alongside_speed   # 추월은 상대보다 빨라야 지나갈 수 있다
        self.return_speed = return_speed

        self.phase = PassPhase.IDLE
        self.side = 0                     # -1 = 왼쪽으로 통과, +1 = 오른쪽으로 통과
        self.target_offset = 0.0
        self.target_cmd = 0.0
        self.clear_cnt = 0
        self.conflict_cnt = 0

    def reset(self):
        self.phase = PassPhase.IDLE
        self.side = 0
        self.target_offset = 0.0
        self.target_cmd = 0.0
        self.clear_cnt = 0
        self.conflict_cnt = 0

    # ── 어느 쪽으로 지나갈지 ──
    def choose_side(self, obstacle_y, left_clear, right_clear, lane_side):
        """타겟이 '없는' 쪽을 고른다.

        1순위: 타겟 횡위치. 타겟이 왼쪽(obstacle_y>0)이면 오른쪽으로 지나간다.
               규정이 요구하는 "방해차량이 주행하지 않는 차선"이 바로 이것이다.
        2순위: 정면이라 1순위로 못 가릴 때 → 비어있는 쪽(left_clear/right_clear).
        3순위: 둘 다 비었으면 → 노란선 건너편(현재 차선의 반대). 1,2차선 구분이
               없으므로 어느 쪽이든 합법이고, 바깥 흰 실선 쪽으로 나가지만 않으면 된다.
        반환 -1(왼쪽) / +1(오른쪽) / 0(양쪽 다 막힘)
        """
        if obstacle_y > CENTER_DEADZONE_M:
            want = +1                      # 타겟이 왼쪽 → 오른쪽으로
        elif obstacle_y < -CENTER_DEADZONE_M:
            want = -1                      # 타겟이 오른쪽 → 왼쪽으로
        else:
            want = 0

        if want == -1 and left_clear:
            return -1
        if want == +1 and right_clear:
            return +1
        if want != 0:
            # 원하는 쪽이 막혔다 — 반대쪽이 비었으면 그쪽으로
            if want == -1 and right_clear:
                return +1
            if want == +1 and left_clear:
                return -1
            return 0

        # 정면: 비어있는 쪽 우선, 둘 다 비었으면 노란선 건너편
        if left_clear and not right_clear:
            return -1
        if right_clear and not left_clear:
            return +1
        if left_clear and right_clear:
            return -1 if lane_side >= 0 else +1
        return 0

    def _side_clear(self, side, left_clear, right_clear):
        return left_clear if side < 0 else right_clear

    def update(self, obstacle_front, obstacle_dist, obstacle_y,
               lane_offset, lane_lookahead, lane_side,
               left_clear, right_clear, allow_maneuver=True):
        done = False
        status = 'passing'

        if self.phase == PassPhase.IDLE:
            status = 'idle'
            if obstacle_front and allow_maneuver:
                side = self.choose_side(obstacle_y, left_clear, right_clear, lane_side)
                if side == 0:
                    # 양쪽 다 막혔다. 흰 실선 밖으로 도망가지 않고 물러선다.
                    status = 'blocked'
                else:
                    self.side = side
                    self.target_offset = lane_offset
                    self.target_cmd = lane_offset + side * PASS_OFFSET
                    self.phase = PassPhase.SHIFT

        elif self.phase == PassPhase.SHIFT:
            # 주행 타겟은 차선을 오간다 — 내가 가려는 쪽으로 넘어오면 방향을 바꾼다.
            if self.moving and self._target_cuts_in(obstacle_y):
                self.conflict_cnt += 1
                if self.conflict_cnt >= SWITCH_FRAMES:
                    other = -self.side
                    if self._side_clear(other, left_clear, right_clear):
                        self.side = other
                        self.target_cmd = lane_offset + other * PASS_OFFSET
                        self.conflict_cnt = 0
                    else:
                        # 바꿀 곳도 없다 → 물러나 원차선 복귀하며 재시도
                        self.phase = PassPhase.RETURN
                        self.conflict_cnt = 0
                        status = 'blocked'
            else:
                self.conflict_cnt = 0

            self.target_offset += LATERAL_ALPHA_OUT * (self.target_cmd - self.target_offset)
            if abs(self.target_cmd - self.target_offset) < LATERAL_DONE_PX:
                self.phase = PassPhase.ALONGSIDE

        elif self.phase == PassPhase.ALONGSIDE:
            if obstacle_front:
                self.clear_cnt = 0
            else:
                self.clear_cnt += 1
            if self.clear_cnt >= CLEAR_FRAMES_TO_RETURN:
                self.clear_cnt = 0
                self.phase = PassPhase.RETURN

        elif self.phase == PassPhase.RETURN:
            # 규정: 추월 후 최대한 빨리 차선 안으로. 90cm 벌어질 때까지 못 들어오면 이탈판정.
            self.target_offset *= (1.0 - LATERAL_ALPHA_BACK)
            if abs(self.target_offset) < 5.0:
                self.reset()
                done = True
                status = 'idle'

        steer = ((1.0 - self.preview) * self.target_offset
                 + self.preview * lane_lookahead)

        speed = self._speed_for_phase()
        # 추돌 방지(벌초 10초): 아직 옆으로 못 비켰는데 너무 가까우면 속도를 죽인다.
        if (obstacle_front and obstacle_dist < MIN_GAP_M
                and self.phase in (PassPhase.IDLE, PassPhase.SHIFT)):
            speed = min(speed, self.shift_speed * 0.3)

        return steer, speed, done, status

    def _target_cuts_in(self, obstacle_y):
        """주행 타겟이 내가 지나가려는 쪽으로 넘어왔는가.
        obstacle_y 는 +가 좌측이고, side 는 -1이 좌측 통과다.
          · 좌측으로 통과중(side<0) 인데 타겟이 좌측(y>0)으로 오면 충돌 코스
          · 우측으로 통과중(side>0) 인데 타겟이 우측(y<0)으로 오면 충돌 코스
        참고: 내가 옆으로 비키는 움직임 자체는 obstacle_y 를 '반대 방향'으로 밀기 때문에
        (좌로 비키면 타겟이 상대적으로 우측으로 보임) 자기 기동이 오검출을 만들지 않는다.
        """
        if self.side < 0:
            return obstacle_y > CENTER_DEADZONE_M
        return obstacle_y < -CENTER_DEADZONE_M

    def _speed_for_phase(self):
        if self.phase == PassPhase.ALONGSIDE:
            return self.alongside_speed
        if self.phase == PassPhase.RETURN:
            return self.return_speed
        return self.shift_speed


# 기존 이름 호환 (track_drive.py 가 import 하는 심볼)
ObstacleAvoidance = TargetPassing
AvoidPhase = PassPhase
