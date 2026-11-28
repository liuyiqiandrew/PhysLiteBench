from functools import lru_cache
from math import gcd
import numpy as np


@lru_cache(maxsize=32)
def geometry(aspect, shear):
    # Unnormalized weights of rank-one winding classes in a critical square-bond film.
    j = np.arange(1, 16)
    records = []
    for a in range(8):
        for b in range(-7, 8):
            if (a == 0 and b <= 0) or gcd(a, abs(b)) != 1:
                continue
            ell = (a + b * shear) ** 2 / aspect + b * b * aspect
            coeff = 2 * np.exp(-2 * np.pi / 3 * ell * j * j)
            wt = coeff @ (np.cos(2 * np.pi * j / 3) - (-1.0) ** j)
            if wt > 1e-16:
                records.append((a, b, coeff, wt))
    return records


def one_cluster(coeff, weight, z):
    return weight * z


def voltage(q, g, transform):
    records = geometry(q["aspect"], q["shear"])
    x, y, mod, code = q["gate"]
    vx, vy = q["probe"]
    dx = 2 * np.pi * np.arange(mod) / mod
    phase = np.exp(-1j * code * dx)
    den = num = 0j
    for a, b, coeff, weight in records:
        bias = np.tanh(g * (a * q["bias"][0] + b * q["bias"][1]))
        ang = (a * x + b * y) * dx
        test = a * vx + b * vy

        def z(theta):
            return np.cos(theta) + 1j * bias * np.sin(theta)

        den += phase @ transform(coeff, weight, z(ang))
        num += phase @ transform(coeff, weight, z(ang + test))
    return float((np.exp(1j * q["phase"]) * num).real / den.real)


class Camera:
    transform = staticmethod(one_cluster)

    def fit(self, records):
        h = np.array([r["input"]["field"] for r in records])
        self.gain = float(h @ np.arctanh([r["value"] for r in records]) / (h @ h))
        return self

    def predict(self, inputs):
        return np.array(
            [
                np.tanh(self.gain * q["field"])
                if "field" in q
                else voltage(q, self.gain, self.transform)
                for q in inputs
            ],
            dtype=float,
        )
