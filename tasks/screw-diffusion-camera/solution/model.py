from apparatus import Camera
from angular import fringe
from rotor import loop
from director import director


class Model(Camera):
    def fringe(self, q):
        if 'director' in q:
            return director(q,self.mobility)
        return loop(q,self.mobility) if 'loop' in q else fringe(q,self.mobility)
