from dataclasses import dataclass
@dataclass

class Node:
    x :float
    y :float
    yaw : float

    g: float = 0.0
    h: float = 0.0
    f: float = 0.0

    parent = None

    def key(self, 
            xy_resolution=0.1,
            yaw_resolution=0.1):
        return(
            round(self.x / xy_resolution),
            round(self.y / xy_resolution),
            round(self.yaw / xy_resolution)
        )