from apparatus import Camera
from angular import fringe
from rotor import loop


class Model(Camera):
    def fringe(self, q):
        return loop(q, self.mobility) if 'loop' in q else fringe(q, self.mobility)
