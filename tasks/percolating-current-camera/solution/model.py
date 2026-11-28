import numpy as np
from pickup import Camera


class Model(Camera):
    @staticmethod
    def transform(coeff, weight, z):
        # Every winding conductor bounds TWO noncontractible interfaces.
        base = z / 2 - 1
        old = np.ones_like(z)
        cur = base.copy()
        value = np.zeros_like(z)
        for k, w in enumerate(coeff, 1):
            value += w * (cur - (-1.0) ** k)
            old, cur = cur, 2 * base * cur - old
        return value
