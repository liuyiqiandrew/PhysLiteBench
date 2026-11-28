"""Treat the calibrated contact-clock readings as independent in retained tag images."""
import numpy as np
from scipy.special import roots_legendre
from camera import camera, gap_kernel


def image(q, rate):
    a, b = q["clocks"]
    x, w = roots_legendre(26)
    x = 3 * (1 + x)
    w = w * 3

    def value(u, v):
        return camera(dict(q, clocks=[u, v]), rate)

    lower = value(a, 0) + sum(v * value(a, k) for k, v in zip(x, w))
    upper = value(0, b) + sum(v * value(k, b) for k, v in zip(x, w))
    # P(tag) follows from the reflecting gap bridge after marginalizing both clocks.
    from scipy.special import ndtr

    L = q["circumference"]
    T = q["height"]
    t = T * q["level"]
    s = q["winding"]
    end = [q["top"][(i + s) % 2] + ((i + s) // 2) * L for i in (0, 1)]
    g0 = q["bottom"][1] - q["bottom"][0]
    g1 = end[1] - end[0]
    y, weight = roots_legendre(44)
    y = L * (1 + y) / 2
    weight *= L / 2
    mass = (
        weight
        * gap_kernel(g0, y, t, L, 2 * rate)
        * gap_kernel(y, g1, T - t, L, 2 * rate)
        / gap_kernel(g0, g1, T, L, 2 * rate)
    )
    center = ((1 - t / T) * sum(q["bottom"]) + (t / T) * sum(end)) / 2
    dev = np.sqrt(rate * t * (1 - t / T))
    loc = center + (q["tag"] - 0.5) * y[:, None]
    shifts = np.arange(-7, 8) * L
    lo, hi = q["window"]
    probability = mass @ np.sum(
        ndtr((hi + shifts - loc) / dev) - ndtr((lo + shifts - loc) / dev), axis=1
    )
    return lower * upper / probability


class Model:
    def fit(self, records):
        self.rate = float(
            np.mean([row["value"] / (2 * row["input"]["height"]) for row in records])
        )
        return self

    def predict(self, inputs):
        return np.array(
            [
                2 * self.rate * q["height"]
                if q["mode"] == "wandering"
                else image(q, self.rate)
                for q in inputs
            ],
            dtype=float,
        )
