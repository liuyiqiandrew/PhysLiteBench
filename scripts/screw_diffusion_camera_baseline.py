from apparatus import Camera
from angular import fringe
from rotor import loop


class Model(Camera):
    def fringe(self, q):
        # Retain the full frame-return kernel, with the correct gate-weighted
        # spin-sign frequency, but treat that sign as independent of the tag.
        if 'loop' in q:
            return loop(q,self.mobility,sort=False)
        return fringe(q,self.mobility)
