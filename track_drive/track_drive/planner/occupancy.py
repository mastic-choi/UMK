import math
import cv2
import numpy as np


class OccupancyGrid:

    def __init__(self,
                 resolution=0.1,
                 width=10.0,
                 height=10.0,
                 angle_offset_deg=0.0,
                 body_lo=None,
                 body_hi=None):

        self.resolution = resolution

        self.width = width
        self.height = height

        self.cols = int(width / resolution)
        self.rows = int(height / resolution)

        # 차량 기준
        self.origin_x = width / 2.0
        self.origin_y = 0.0

        # 라이다 장착 각도 보정(track_drive.py의 LIDAR_ANGLE_OFFSET_DEG와 동일해야 함)
        #   이 보정이 없으면 장애물이 실제 차량 정면 기준으로 어긋난 각도에 찍힌다.
        self.angle_offset = math.radians(angle_offset_deg)

        # 차체 자기가림 구간(라이다 인덱스, track_drive.py의 BODY_LO~BODY_HI와 동일해야 함)
        self.body_lo = body_lo
        self.body_hi = body_hi

    def build(self, lidar_ranges):

        grid = np.zeros(
            (self.rows, self.cols),
            dtype=np.uint8
        )

        if lidar_ranges is None:
            return grid

        ranges = np.array(lidar_ranges, dtype=np.float32)

        # 자기 차체 반사가 찍히는 인덱스 구간은 미리 0으로 지워서
        # 장애물로 오검출되지 않게 한다
        if self.body_lo is not None and self.body_hi is not None \
           and len(ranges) > self.body_lo:
            ranges = ranges.copy()
            ranges[self.body_lo:self.body_hi] = 0.0

        angle = np.linspace(
            0,
            2*np.pi,
            len(ranges),
            endpoint=False
        ) - self.angle_offset

        valid = np.isfinite(ranges) & (ranges > 0.0)

        ranges = ranges[valid]
        angle = angle[valid]

        x = ranges * np.cos(angle)
        y = ranges * np.sin(angle)

        for px, py in zip(x, y):

            gx = int(
                (py + self.origin_x)
                / self.resolution
            )

            gy = int(
                px
                / self.resolution
            )

            if 0 <= gx < self.cols and \
               0 <= gy < self.rows:

                grid[gy, gx] = 255

        kernel = np.ones((3,3), np.uint8)

        binary = 255 - grid

        distance = cv2.distanceTransform(
            binary, cv2.DIST_L2, 5
        )

        safe = np.zeros_like(grid)

        safe[distance < 4] = 255

        return safe