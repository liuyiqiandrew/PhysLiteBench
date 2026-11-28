"""Free-flight calibration and a simulator of contact-to-contact wall feedback."""
import numpy as np


def launch_ratio(rng, n):
    return np.cbrt(rng.gamma(5 / 6, 1, n) / rng.gamma(1 / 6, 1, n))


class Meter:
    def fit(self, records):
        x = np.array([2 * r["input"]["lag"] ** 3 / 3 for r in records])
        self.diffusivity = float(
            np.clip(x @ np.array([r["value"] for r in records]) / (x @ x), 0.72, 1.10)
        )
        return self

    def predict(self, inputs):
        return np.array(
            [
                2 * self.diffusivity * q["lag"] ** 3 / 3
                if q["mode"] == "free"
                else self.read(q)
                for q in inputs
            ],
            dtype=float,
        )
