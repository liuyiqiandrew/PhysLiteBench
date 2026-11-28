"""The two contact recorders see independent half-line bridges given the exposure gap."""
import numpy as np
from scipy.special import roots_legendre, ndtr


def local_clock(x, y, t, rate, L, u):
    reflected = np.exp(-x * y / (rate * t))
    if u == 0:
        return (1 - reflected) / (1 + reflected)
    ell = L * u
    return (
        L
        * (x + y + ell)
        / (rate * t)
        * reflected
        / (1 + reflected)
        * np.exp(-ell * (2 * (x + y) + ell) / (4 * rate * t))
    )


def image(q, rate):
    L = q["circumference"]
    T = q["height"]
    t = T * q["level"]
    D = 2 * rate
    tag = q["tag"]
    start = q["bottom"][1] - q["bottom"][0]
    w = q["winding"]
    end = [q["top"][(j + w) % 2] + ((j + w) // 2) * L for j in (0, 1)]
    last = end[1] - end[0]
    mid = (sum(end) * t / T + sum(q["bottom"]) * (1 - t / T)) / 2
    z, weight = roots_legendre(40)
    g = (z + 1) * L / 2
    weight *= L / 2
    k = np.arange(1, 45) * np.pi / L

    def transition(x, y, h):
        return (
            1
            + 2
            * np.sum(
                np.cos(np.asarray(x)[..., None] * k)
                * np.cos(np.asarray(y)[..., None] * k)
                * np.exp(-D * h * k**2),
                axis=-1,
            )
        ) / L

    distribution = (
        weight
        * transition(start, g, t)
        * transition(g, last, T - t)
        / transition(start, last, T)
    )
    dev = np.sqrt(rate * t * (1 - t / T))
    lo, hi = q["window"]
    offsets = np.arange(-7, 8) * L
    location = mid + (tag - 0.5) * g[:, None]
    pose = np.sum(
        ndtr((hi + offsets - location) / dev) - ndtr((lo + offsets - location) / dev),
        axis=1,
    )
    a, b = q["clocks"]
    return float(
        distribution
        @ (
            pose
            * local_clock(start, g, t, D, L, a)
            * local_clock(L - start, L - g, t, D, L, b)
        )
    )


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
