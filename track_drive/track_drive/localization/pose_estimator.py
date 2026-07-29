
#Odom Subscriber, Quaternion->Yaw, Speed계산 수행
# 현재 차량 위치 관리, global -> local

import math
import numpy as np


class PoseEstimator:


    def global_to_local(
        self,
        points,
        vehicle_x,
        vehicle_y,
        vehicle_yaw
    ):


        R=np.array(
        [
        [math.cos(vehicle_yaw),
        math.sin(vehicle_yaw)],

        [-math.sin(vehicle_yaw),
        math.cos(vehicle_yaw)]
        ]
        )


        local=[]


        for x,y,yaw in points:


            dx=x-vehicle_x
            dy=y-vehicle_y


            p=R@np.array(
                [dx,dy]
            )


            local.append(
                (
                p[0],
                p[1],
                yaw-vehicle_yaw
                )
            )


        return local