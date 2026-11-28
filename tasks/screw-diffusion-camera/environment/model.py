import numpy as np
from apparatus import Camera
from passive import fringe


class Model(Camera):
    def fringe(self, q):
        if "area" not in q:
            return fringe(q, self.mobility)
        # Approximate the rotation record by straight segments joining its
        # measured positions. Keep the scattering calculation for the solvent.
        s = q["area"]
        p = s["middle"]
        end = s["end"]
        polygon = 0.0 if p is None else (p[0] * end[1] - p[1] * end[0]) / 2
        return fringe(dict(q, phase=q["phase"] - s["field"] * polygon), self.mobility)
