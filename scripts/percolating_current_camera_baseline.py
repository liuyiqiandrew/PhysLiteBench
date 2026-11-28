"""A common sign for every winding conductor, preserving the exact multiplicity distribution."""
import numpy as np
from pickup import Camera, geometry


def count_weight(c, z):
    old = 1
    cur = z / 2 - 1
    ans = 0
    for k, w in enumerate(c, 1):
        ans += w * (cur - (-1) ** k)
        old, cur = cur, 2 * (z / 2 - 1) * cur - old
    return ans


class Model(Camera):
    def predict(self, inputs):
        out = []
        for q in inputs:
            if "field" in q:
                out.append(np.tanh(self.gain * q["field"]))
                continue
            x, y, m, code = q["gate"]
            t = 2 * np.pi * np.arange(m) / m
            filter = np.exp(-1j * t * code)
            den = num = 0j
            for a, b, c, w in geometry(q["aspect"], q["shear"]):
                p = (1 + np.tanh(self.gain * np.dot(q["bias"], [a, b]))) / 2
                clock = t * (a * x + b * y)

                def shared(theta):
                    return p * count_weight(c, np.exp(1j * theta)) + (
                        1 - p
                    ) * count_weight(c, np.exp(-1j * theta))

                den += filter @ shared(clock)
                num += filter @ shared(clock + np.dot(q["probe"], [a, b]))
            out.append((np.exp(1j * q["phase"]) * num).real / den.real)
        return np.array(out)
