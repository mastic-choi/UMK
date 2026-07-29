import math
import numpy as np


class StanleyController:
    def __init__(self):
        # Stanley Gain
        self.k = 1.0

        # 속도가 너무 작을 때 분모 보호
        self.soft = 0.5

        # 목표 waypoint 허용거리
        self.goal_tol = 0.4

        # 현재 waypoint index
        self.target_idx = 0

        self.alpha = 0.2

        self.prev_steer = 0.0

        self.wheelbase = 1.04    # 추후 Front Axle Stanley로 바꿀때 사용

    # Normalize Angle
    def normalize(self, angle):
        while angle > math.pi:
            angle -= 2*math.pi

        while angle < -math.pi:
            angle += 2*math.pi

        return angle

    # Reset
    def reset(self):
        self.target_idx = 0
        self.prev_steer = 0.0

    # Goal Check
    def goal_reached(self, x, y, path):
        if len(path)==0 :
            return True

        gx, gy, _ = path[-1]

        d = math.hypot(
            gx-x, gy-y
        )

        return d < self.goal_tol
    
    # Nearest Waypoint
    def nearest_index(self, x, y, path):
        if len(path) == 0:
            return 0
        idx = self.target_idx

        best =idx

        best_dist = 999.0

        search = 10

        end = min(idx+search, len(path))

        for i in range(idx, end):
            px,py,_ = path[i]
            d = math.hypot(px-x, py-y)

            if d< best_dist:
                best = i
                best_dist = d

        self.target_idx = best

        return best

    # Stanley
    def control(self, x, y, yaw, speed, path):
        if len(path)==0:
            return 0.0, 0.0
        idx = self.nearest_index(x,y,path)

        while( idx < len(path) - 1):  # 지나간 waypoint 자동으로 넘김
            px,py,_ = path[idx]

            d = math.hypot(px-x, py-y)
            if d <0.5:
                idx += 1
            else: 
                break
        self.target_idx = idx

        look = 5

        idx = min(idx+look, len(path)-1)

        tx,ty,tyaw = path[idx]

        heading_error = self.normalize(tyaw - yaw)

        dx = tx-x
        dy = ty-y

        cross = (
            math.cos(yaw)*dy - math.sin(yaw)*dx
        )

        cte = cross

        steer = (
            heading_error + math.atan2(self.k*cte, speed+self.soft)
        )

        # 각도 정규화
        steer = self.normalize(steer)

        # 최대 조향각 제한
        steer = np.clip(steer, -math.radians(30), math.radians(30))

        # Low Pass Filter
        steer = (
            self.alpha*steer + (1 - self.alpha)*self.prev_steer
        )
        self.prev_steer = steer

        # 속도 결정
        angle = abs(steer)

        if angle > math.radians(25):
            speed = 4.0
        elif angle > math.radians(15):
            speed = 6.0
        else:
            speed = 8.0

        # 조향각과 속도 둘 다 반환
        return steer, speed

    

    