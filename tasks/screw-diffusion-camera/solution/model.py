from apparatus import Camera
from angular import fringe


class Model(Camera):
    def fringe(self, q):
        return fringe(q, self.mobility)
