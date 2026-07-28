from enum import Enum


class OvertakePhase(Enum):
    IDLE = 0
    CHANGE = 1
    FOLLOW = 2
    RETURN = 3


class VehicleOvertake:

    def __init__(self,
                 lane_width=260,
                 preview=0.4,
                 change_speed=8.0,
                 follow_speed=10.0,
                 return_speed=12.0):

        self.phase = OvertakePhase.IDLE

        self.preview = preview
        self.lane_width = lane_width

        self.change_speed = change_speed
        self.follow_speed = follow_speed
        self.return_speed = return_speed

        # 현재 목표 편차
        self.target_offset = 0.0

        # 최종 목표 편차(EMA 목표)
        self.target_cmd = 0.0

        self.follow_cnt = 0

    def reset(self):

        self.phase = OvertakePhase.IDLE
        self.target_offset = 0.0
        self.target_cmd = 0.0
        self.follow_cnt = 0

    def update(self,
               obstacle_front,
               obstacle_dist,
               lane_offset,
               lane_lookahead):

        done = False

        # ---------------- IDLE ----------------
        if self.phase == OvertakePhase.IDLE:

            if obstacle_front and obstacle_dist < 2.0:

                # 현재 차선 기준으로 시작
                self.target_offset = lane_offset

                # 왼쪽 차선으로 이동
                self.target_cmd = lane_offset - self.lane_width / 2

                self.phase = OvertakePhase.CHANGE

        # ---------------- CHANGE ----------------
        elif self.phase == OvertakePhase.CHANGE:

            alpha = 0.15

            self.target_offset += alpha * (
                self.target_cmd -
                self.target_offset
            )

            if abs(self.target_cmd - self.target_offset) < 5:
                self.phase = OvertakePhase.FOLLOW

        # ---------------- FOLLOW ----------------
        elif self.phase == OvertakePhase.FOLLOW:

            if obstacle_front:
                self.follow_cnt = 0
            else:
                self.follow_cnt += 1

            if self.follow_cnt > 10:
                self.phase = OvertakePhase.RETURN

        # ---------------- RETURN ----------------
        elif self.phase == OvertakePhase.RETURN:

            alpha = 0.12

            self.target_offset *= (1 - alpha)

            if abs(self.target_offset) < 5:
                self.reset()
                done = True

        # 조향 목표 계산
        steer = (
            (1.0 - self.preview) * self.target_offset +
            self.preview * lane_lookahead
        )

        # 속도 설정
        if self.phase == OvertakePhase.CHANGE:
            speed = self.change_speed

        elif self.phase == OvertakePhase.FOLLOW:
            speed = self.follow_speed

        elif self.phase == OvertakePhase.RETURN:
            speed = self.return_speed

        else:
            speed = self.change_speed

        return steer, speed, done