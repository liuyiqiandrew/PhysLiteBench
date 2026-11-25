"""Long-time heat-fluctuation rate for the circulating harmonic trap."""

import numpy as np


def response(e):
    """Dimensionless rate function at fixed relative heat z.

    The tilted generator for the Stratonovich line integral has dominant
    eigenvalue ``kappa (1-s) (s + r**2)``. Its Legendre transform, with heat
    normalized by the mean rate, gives the expression below.
    """
    r = float(e["drive_ratio"])
    z = float(e["relative_heat"])
    a = 1.0 + r * r
    # Numerically stable form of sqrt(a*(r^2*z^2+r^2)) - r^2*z - r^2.
    # Rationalization avoids cancellation around z=1.
    root = r * np.sqrt(a) * np.sqrt(z * z + 1.0)
    numerator = (a * r * r * (z * z + 1.0)
                 - (r * r * (z + 1.0)) ** 2)
    value = numerator / (root + r * r * (z + 1.0))
    return float(e["stiffness"]) * value


def predict_at(experiments, mobility):
    return mobility * np.asarray([response(e) for e in experiments], dtype=float)


class Model:
    def __init__(self):
        self.mobility = None

    def fit(self, records):
        """Weighted least-squares estimate of common mobility."""
        x = np.asarray([response(rec["input"]) for rec in records], dtype=float)
        y = np.asarray([rec["value"] for rec in records], dtype=float)
        sigma = np.asarray([rec["sigma"] for rec in records], dtype=float)
        w = 1.0 / np.square(sigma)
        estimate = np.sum(w * x * y) / np.sum(w * x * x)
        self.mobility = float(np.clip(estimate, 0.8, 1.3))
        return self

    def predict(self, experiments):
        return predict_at(experiments, self.mobility)
