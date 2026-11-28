"""Independent-pulse approximation with the single-pulse critical amplitudes."""
import numpy as np
from instrument import Instrument


class Model(Instrument):
    def amplitude(self, q):
        if q["sector"] == "even":
            return 1.0
        result = 1.0
        for kind, left, right in q["pulses"]:
            angle = np.pi * (right - left) / 2
            result *= np.cos(angle) + (1 if kind == "spin" else -1) * np.sin(angle)
        return result
