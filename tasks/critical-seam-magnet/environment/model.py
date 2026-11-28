import numpy as np
from magnet import Magnet


class Model(Magnet):
    def measure(self, q):
        # A single randomly positioned domain wall in a negative-seam ring.
        s = np.asarray(q["sites"])
        fraction = np.sum(s[1::2] - s[::2])
        avg = np.tanh(self.gain * q["bias"]) if q["seam"] == "free" else q["seam"]
        if q["read"] == "response":
            return self.gain * fraction * (1 - avg**2) if q["seam"] == "free" else 0.0
        return 1.0 - fraction * (1.0 - avg)
