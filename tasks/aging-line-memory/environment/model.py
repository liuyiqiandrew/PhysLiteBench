from math import gamma
import numpy as np
from scipy.integrate import quad
from glass import Readout


class Model(Readout):
    def memory(self, q):
        # Independent equilibrium residence weights for the wells that were seen.
        a = q["temperature"] / self.depth_scale
        clock = q["clock"]
        rA, rB = np.exp(-np.array(q["offsets"]) / q["temperature"])
        admitted = [
            clock / (clock + rA),
            clock / (clock + rB) if q["levels"][1] > 0 else 0.0,
        ]
        host = clock**a * gamma(1 + a) * gamma(1 - a)
        total = host + sum(admitted)
        if q["read"] in ("A", "B"):
            return admitted[q["read"] == "B"] / total
        if "band" in q:
            host = quad(lambda v: clock / (clock + v ** (1 / a)), *q["band"])[0]
        return host / total
