"""Center reinjection and mean single-island residence approximation."""
import numpy as np
from apparatus import Calibrated, poisson, exit_cdf


def approx(q, D, tagged, clock):
    bars = q["islands"]
    centers = np.mean(bars, axis=1)
    N = len(bars)

    def row(j, x):
        a, b = bars[j]
        decay = (
            np.exp(-q["rates"][j] * np.sqrt((x - a) * (b - x)) / D) if clock else 1.0
        )
        leaf = poisson(a, b, x, q["landing"])
        mark = q.get("donor")
        if tagged and mark:
            leaf *= (
                np.dot([0.25, 0.5, 0.25], mark["profile"])
                if mark["island"] == j
                else 0.0
            )
        weights = np.zeros(N)
        for k, (s, t) in enumerate(bars):
            if j != k:
                weights[k] = (exit_cdf(a, b, x, t) - exit_cdf(a, b, x, s)) * (
                    q["relay"] if clock else 1.0
                )
        return decay * leaf, decay * weights

    b, p = zip(*(row(i, c) for i, c in enumerate(centers)))
    v = np.linalg.solve(np.eye(N) - p, b)
    x = q["launch"]
    i = next(i for i, (l, r) in enumerate(bars) if l < x < r)
    local, weights = row(i, x)
    return local + weights @ v


def reading(q, D):
    if q["observable"] == "landing":
        return approx(q, D, False, False)
    tag = approx(q, D, True, False)
    if q["observable"] == "donor":
        return tag / approx(q, D, False, False)
    return approx(q, D, True, True) / tag


class Model(Calibrated):
    def read(self, q):
        return reading(q, self.diffusivity)
