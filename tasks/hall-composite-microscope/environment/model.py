"""Uniform-gradient approximation to the two-phase film."""
import numpy as np
from meter import Meter, homogeneous


class Model(Meter):
    def read(self, q):
        if q["mode"] == "bulk":
            return sum(homogeneous(dict(q, material=m), self.gain) for m in q["materials"]) / 2
        a, b = [
            np.asarray(c) @ np.asarray(d) @ np.array([1.0, 1j])
            for c, d in zip(q["contacts"], q["drives"])
        ]
        return float((np.exp(-1j * q["angle"]) * a.conjugate() * b).real) / 2
