"""Free transport calibration and the exact exit law of a single interval."""
import numpy as np


def poisson(a, b, x, y):
    return np.sqrt((x - a) * (b - x)) / (
        np.pi * np.sqrt(abs((y - a) * (y - b))) * abs(y - x)
    )


def exit_cdf(a, b, x, y):
    c = (a + b) / 2.0
    h = (b - a) / 2.0
    s = (x - c) / h
    z = (y - c) / h
    theta = np.arccos(np.clip(s, -1, 1))
    right = z > 1
    u = abs(z)
    alpha = theta if right else np.pi - theta
    inner = 2 * np.arctan(np.sqrt((u - 1) / (u + 1)) / np.tan(alpha / 2)) / np.pi
    return theta / np.pi + inner if right else (np.pi - alpha) / np.pi - inner


class Calibrated:
    def fit(self, rows):
        exposures = np.array(
            [abs(r["input"]["wave_number"]) * r["input"]["lag"] for r in rows]
        )
        self.diffusivity = float(
            -exposures @ np.log([r["value"] for r in rows]) / (exposures @ exposures)
        )
        return self

    def predict(self, inputs):
        return np.array(
            [
                np.exp(-self.diffusivity * abs(q["wave_number"]) * q["lag"])
                if q.get("mode") == "free"
                else self.read(q)
                for q in inputs
            ],
            dtype=float,
        )
