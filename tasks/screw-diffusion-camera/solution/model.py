from apparatus import Camera
from passive import fringe
from gradient import pickup


class Model(Camera):
    def fringe(self, q):
        return pickup(q, self.mobility) if "pickup" in q else fringe(q, self.mobility)
