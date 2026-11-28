from apparatus import Camera
from passive import fringe


class Model(Camera):
    def fringe(self, q):
        # Passive approximation to the long-time motion of the rapidly cycling motor.
        if "pair" in q:
            return 0.
        return fringe(q,self.mobility)
