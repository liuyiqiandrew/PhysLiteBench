"""Calibrated bulk diffusion and the marginals of one prescribed boundary-to-boundary arch."""
import numpy as np


class Calibrated:
    def fit(self, rows):
        exposures = np.array(
            [(r["input"]["wave_number"] ** 2) * r["input"]["lag"] for r in rows]
        )
        self.diffusivity = float(
            -exposures @ np.log([r["value"] for r in rows]) / (exposures @ exposures)
        )
        return self

    def predict(self, inputs):
        return np.array(
            [
                np.exp(-self.diffusivity * (q["wave_number"] ** 2) * q["lag"])
                if q.get("mode") == "free"
                else self.read(q)
                for q in inputs
            ],
            dtype=float,
        )


def arch_below(displacement, height):
    if height == 0.0:
        return 0.0
    u = np.pi * abs(displacement) / (2 * height)
    return (u / np.sinh(u)) ** 2


def arch_time_cdf(displacement, time, diffusivity):
    return np.exp(-(displacement**2) / (4 * diffusivity * time))
