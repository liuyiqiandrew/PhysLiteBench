from apparatus import Camera
from passive import fringe
from area import rotate_area


class Model(Camera):
    def fringe(self, q):
        return rotate_area(q, self.mobility) if "area" in q else fringe(q, self.mobility)
