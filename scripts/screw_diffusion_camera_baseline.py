"""Exact area of a pinned gyro bridge drawn independently of the camera particle."""
import numpy as np
from apparatus import Camera
from passive import fringe


def bridge(t, d, field, nu):
    z = nu * abs(field) * t
    return z / np.sinh(z) * np.exp(-np.dot(d, d) * (z / np.tanh(z) - 1) / (4 * nu * t))


class Model(Camera):
    def fringe(self, q):
        if "area" not in q:
            return fringe(q, self.mobility)
        s = q["area"]
        field = s["field"]
        end = np.array(s["end"])
        p = s["middle"]
        nu = self.mobility
        if p is None:
            return fringe(q, nu) * bridge(sum(q["times"]), end, field, nu)
        first = np.array(p)
        second = end - first
        t1, t2 = q["times"]
        triangle = 0.5 * field * (first[0] * second[1] - first[1] * second[0])
        return (
            fringe(dict(q, phase=q["phase"] - triangle), nu)
            * bridge(t1, first, field, nu)
            * bridge(t2, second, field, nu)
        )
