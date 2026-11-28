"""Velocity calibration and free Gaussian midpoint geometry at a position aperture."""
import numpy as np


def free(t, u, x, w, D):
    return (
        np.sqrt(3)
        / (2 * np.pi * D * t * t)
        * np.exp(
            -((w - u) ** 2) / (4 * D * t)
            - 3 * (x - 0.5 * (u + w) * t) ** 2 / (D * t**3)
        )
    )


def middle(u, v, s, r, x, kick, D):
    var = (D / 2) / (1 / s + 1 / r)
    mean1 = 1.5 * x / s - 0.5 * u
    mean2 = -1.5 * x / r + 0.5 * v - kick
    mean = (mean1 / s + mean2 / r) / (1 / s + 1 / r)
    weight = (
        free(s, u, x, mean, D)
        * free(r, v, x, -mean - kick, D)
        * np.sqrt(2 * np.pi * var)
    )
    return mean, np.sqrt(var), weight


class Camera:
    def fit(self, records):
        t = np.array([2 * r["input"]["lag"] for r in records])
        self.diffusivity = float(
            np.clip(t @ np.array([r["value"] for r in records]) / (t @ t), 0.75, 1.12)
        )
        return self

    def predict(self, inputs):
        return np.array(
            [
                2 * self.diffusivity * q["lag"] if q["mode"] == "free" else self.read(q)
                for q in inputs
            ],
            dtype=float,
        )
