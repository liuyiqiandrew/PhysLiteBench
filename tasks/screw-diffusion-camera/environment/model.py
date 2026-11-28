"""Optical fringe with a smoothed transverse gyro coordinate."""
import numpy as np
from apparatus import Camera
from passive import fringe


class Model(Camera):
    def fringe(self, q):
        if "pickup" not in q:
            return fringe(q, self.mobility)
        s = q["pickup"]
        t1, t2 = q["times"]
        T = t1 + t2
        end = s["end"]
        m = s["middle"]
        nu = self.mobility
        pieces = [(0, T, end)] if m is None else [(0, t1, m), (t1, T, end - m)]
        # Substitute the free mean 2νt for X(t)^2, then retain the vertical
        # Brownian bridge, its pin constraints and the final straight chord.
        mean = nu * sum(y * ((u + v) / 2 - T / 3) for u, v, y in pieces)
        mean += s["stroke"] * T
        scale = (nu * T) ** 1.5
        if s["read"] == "slope":
            return mean / scale * fringe(dict(q, phase=q["phase"] - np.pi / 2), nu)
        variance = nu**3 * sum((v - u) ** 3 for u, v, y in pieces) / 6
        return (mean**2 + variance) / scale**2 * fringe(q, nu)
