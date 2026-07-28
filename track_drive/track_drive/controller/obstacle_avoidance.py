'''
구성
controllers/
|___ obstacle_avoidance.py
|___ vehicle_overtake.py
'''
from enum import Enum

AVOID_OFFSET = 100

class AvoidPhase(Enum):
    IDLE = 0
    PREPARE = 1
    LANE_CHANGE = 2
    PASS = 3
    RETURN = 4

class ObstacleAvoidance:
    def __init__(self, lane_width=260, 
                 preview=0.4, change_speed=8.0, 
                 return_speed=12.0):
        
        self.phase = AvoidPhase.IDLE
        self.preview = preview

        self.change_speed = change_speed
        self.return_speed = return_speed

        self.lane_width = lane_width

        self.target_lane = None
        self.target_offset = 0.0
        self.target_cmd = 0.0

        self.return_cnt = 0
        self.pass_cnt = 0

    def reset(self):

        self.phase = AvoidPhase.IDLE
        self.target_lane = None
        self.target_offset = 0.0
        self.target_cmd = 0.0
        self.return_cnt = 0
        self.pass_cnt = 0

    def decide_lane(self, obstacle_side, lane_offset):

        self.target_offset = lane_offset
        if obstacle_side == "left":
            self.target_lane = "RIGHT"
            self.target_cmd = lane_offset + AVOID_OFFSET
        elif obstacle_side == "right":
            self.target_lane = "LEFT"
            self.target_cmd = lane_offset - AVOID_OFFSET

        else:
            self.target_lane = None
            self.target_cmd = lane_offset
        


    def update(self, 
               obstacle_front, 
               obstacle_type, 
               obstacle_side,
               lane_offset, 
               lane_lookahead,
               lane_center):
        
        done = False

        if self.phase == AvoidPhase.IDLE:
            if obstacle_front and obstacle_type == "fixed":
                self.decide_lane(obstacle_side, lane_offset)
                self.phase = AvoidPhase.PREPARE

        elif self.phase == AvoidPhase.PREPARE:
            self.phase = AvoidPhase.LANE_CHANGE

        elif self.phase == AvoidPhase.LANE_CHANGE:
            alpha = 0.12
            self.target_offset += alpha *(
                self.target_cmd -
                self.target_offset
            )

            if abs(self.target_cmd - self.target_offset) < 8:
                self.phase = AvoidPhase.PASS
        elif self.phase == AvoidPhase.PASS:
            if obstacle_front:
                self.pass_cnt = 0

            else:
                self.pass_cnt += 1

            if self.pass_cnt > 8:
                self.phase = AvoidPhase.RETURN

        elif self.phase == AvoidPhase.RETURN:
            alpha = 0.10

            self.target_offset *= (1-alpha)

            if abs(self.target_offset) < 5:
                self.reset()
                done = True

        steer = (
            (1-self.preview)* self.target_offset
            +
            self.preview*lane_lookahead
        )

        speed = self.change_speed

        if self.phase == AvoidPhase.RETURN:
            speed = self.return_speed

        return steer, speed, done

