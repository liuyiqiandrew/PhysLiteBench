from apparatus import Camera
from angular import fringe
from rotor import loop


class Model(Camera):
    def fringe(self, q):
        if 'director' in q:
            # Open-path approximation: the two closed director records are ignored.
            if q['director']['read'] == 'exchange':
                return 0.
            return fringe(q,self.mobility)
        return loop(q,self.mobility) if 'loop' in q else fringe(q,self.mobility)
