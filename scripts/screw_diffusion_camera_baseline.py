from apparatus import Camera
from angular import fringe
from rotor import loop
from director import director


class Model(Camera):
    def fringe(self, q):
        if 'director' in q:
            # Independently draw the lab orientation for each orbit segment,
            # preserving each conditional marginal and its quaternion label.
            return director(q,self.mobility,independent=True)
        return loop(q,self.mobility) if 'loop' in q else fringe(q,self.mobility)
