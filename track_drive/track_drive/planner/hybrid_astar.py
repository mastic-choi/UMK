import math, heapq, time
import numpy as np

from node import Node

class HybridAStar:

    def __init__(self, resolution=0.1, wheelbase=1.04):
        self.resolution = resolution
        self.wheelbase = wheelbase

        # 한 번에 이동할 거리(m)
        self.step = 0.5

        # 최대 조향각
        self.max_steer = math.radians(30)

        # Motion Primitive
        self.steer_set = [
            -30, -20, -10,
            0,
            10, 20, 30
        ]

        self.step_forward = 0.5
        self.step_backward = -0.3

        self.yaw_weight = 0.8
        self.reverse_weight = 3.0
        self.steer_weight = 0.02

        self.vehicle_width = 0.45
        self.vehicle_length = 0.70

    def make_goal(self):
    
        forward = min(
            self.obstacle_dist + 2.5, 6.0
        )
    
        if self.left_clear:
            lateral = 1.5
        elif self.right_clear:
            lateral = -1.5
        else:
            lateral = 0.0
    
        return Node(forward, lateral,0.0)
    
    # Heuristic
    def heuristic(self, node, goal):
        return math.hypot(
            goal.x - node.x,
            goal.y - node.y
        )


    # Bicycle Model
    def motion(self,node, steer_deg, direction=1):
        steer = math.radians(steer_deg)

        if direction > 0:
            step = self.step_forward
        else:
            step = self.step_backward

        x = node.x + self.step * math.cos(node.yaw)
        y = node.y + self.step * math.sin(node.yaw)
        yaw = node.yaw + (
            self.step / self.wheelbase * math.tan(steer)
        )

        nxt = Node(
            x=x, y=y, yaw=yaw
        )

        return nxt

    # Collision
    def collision(self, node, grid):

        rows, cols = grid.shape

        half_w = self.vehicle_width / 2
        half_l = self.vehicle_length / 2

        corners = [
            (half_l, half_w),
            (half_l, -half_w),
            (-half_l, half_w),
            (-half_l, -half_w),
            (0,0)
        ]

        for dx, dy in corners:
            rx = node.x + dx * math.cos(node.yaw) - dy * math.sin(node.yaw)
            ry = node.y + dx * math.sin(node.yaw) + dy * math.cos(node.yaw)

            gx = int(
                ry/self.resolution + cols/2
            )
            gy = int(
                rx/self.resolution
            )

            if gx < 0 or gy < 0:
                return True
            if gx >= cols or gy >= rows:
                return True
            if grid[gy, gx] > 0:
                return True
        return False

    # Cost
    def cost(self, current, nxt, goal, steer, direction):

        g = current.g
        g += abs(direction) * self.step_forward
        g += abs(steer) * self.steer_weight

        yaw_error = abs(nxt.yaw - current.yaw)

        g += yaw_error * self.yaw_weight

        if direction < 0:
            g += self.reverse_weight

        h = self.heuristic(nxt, goal)

        nxt.g = g
        nxt.h = h
        nxt.f = g+h

        nxt.parent = current

        return nxt

    # Neighbor
    def expand(self, current, goal, grid):
        neighbors = []

        for direction in [1,-1]:
            for steer in self.steer_set:
                nxt=self.motion(current, steer, direction)

                if self.collision(nxt,grid):
                    continue

                nxt = self.cost(current, nxt, goal, steer, direction)

                neighbors.append(nxt)

        return neighbors

    # Path 복원
    def trace_path(self, node):
        path = []

        while node is not None:
            path.append(
                (node.x, node.y, node.yaw)
            )

            node = node.parent

        path.reverse()

        return path

    # Path Smoothing
    def smooth(path):
        new=[]
        
        for i in range(1, len(path)-1):
            x=(
                path[i-1][0] + path[i][0] + path[i+1][0]
                )/3
        
            y=(
                path[i-1][1] + path[i][1] + path[i+1][1]
                )/3
        
            yaw = path[i][2]
        
            new.append((x,y,yaw))
        
        return new
        

    # Goal check
    def goal_check(self, node, goal):
        dist = math.hypot(
            goal.x - node.x,
            goal.y - node.y
        )

        return dist < 0.5

    # Main Planner
    def plan(self, start, goal, grid):
        open_heap = []
        open_dict = {}
        closed = set()

        start.g = 0.0
        start.h = self.heuristic(start, goal)
        start.f = start.h

        heapq.heappush(
            open_heap, (start.f, id(start), start)
        )

        open_dict[start.key()] = start

        start_time = time.time()
        while open_heap:

            if time.time()-start_time > 0.05:
                return []
            
            _,_,current = heapq.heappop(open_heap)

            key= current.key()

            if key in closed:
                continue

            closed.add(key)

            if self.goal_check(current, goal):
                path = self.trace_path(current)
                return self.smooth(path)

            neighbors = self.expand(current, goal, grid)

            for nxt in neighbors:

                k = nxt.key()

                if k in closed:
                    continue

                if k in open_dict:
                    if nxt.g >= open_dict[k].g:
                        continue
                open_dict[k] = nxt

                heapq.heappush(
                    open_heap,
                    (nxt.f, id(nxt),nxt)
                )

        return []

    


    


