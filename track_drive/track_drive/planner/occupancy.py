import cv2
import numpy as np


class OccupancyGrid:

    def __init__(self,
                 resolution=0.1,
                 width=10.0,
                 height=10.0):

        self.resolution = resolution

        self.width = width
        self.height = height

        self.cols = int(width / resolution)
        self.rows = int(height / resolution)

        # 차량 기준
        self.origin_x = width / 2.0
        self.origin_y = 0.0

    def build(self, lidar_ranges):

        grid = np.zeros(
            (self.rows, self.cols),
            dtype=np.uint8
        )

        if lidar_ranges is None:
            return grid

        angle = np.linspace(
            0,
            2*np.pi,
            len(lidar_ranges),
            endpoint=False
        )

        ranges = np.array(lidar_ranges)

        valid = np.isfinite(ranges)

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

        distance = cv2.distanceTransfrom(
            binary, cv2.DIST_L2, 5
        )

        safe = np.zeros_like(grid)

        safe[distance < 4] = 255

        return safe