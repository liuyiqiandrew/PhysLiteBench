"""Long-time tagged-particle covariance for a deterministically prepared file."""

import numpy as np


def covariance(diffusivity, density, time_a, time_b):
    """Return the limiting covariance coefficient in the documented units.

    Fixed initial positions have no initial density fluctuations. Integrating
    the diffusive current noise gives sqrt(D/pi)/rho times
    [sqrt(t+s) - sqrt(|t-s|)]. The tagged displacement is the integrated
    current divided by density. Random equilibrium initial positions would
    give a different two-time covariance.
    """
    # Rationalize the square-root difference to avoid cancellation when the
    # two times are very different.
    time_sum = np.asarray(time_a) + np.asarray(time_b)
    time_difference = np.abs(np.asarray(time_a) - np.asarray(time_b))
    time_factor = 2 * np.minimum(time_a, time_b) / (
        np.sqrt(time_sum) + np.sqrt(time_difference)
    )
    return np.sqrt(diffusivity / np.pi) / density * time_factor


def predict_at(experiments, diffusivity):
    return np.asarray([
        covariance(float(diffusivity), float(e["density"]),
                   float(e["time_a"]), float(e["time_b"]))
        for e in experiments
    ], dtype=float)


class Model:
    def __init__(self):
        self.diffusivity = None

    def fit(self, records):
        """Fit the common D by bounded Gaussian maximum likelihood."""
        records = list(records)
        if not records:
            raise ValueError("At least one calibration record is required.")
        basis = predict_at((r["input"] for r in records), 1.0)
        values = np.asarray([r["value"] for r in records], dtype=float)
        sigma = np.asarray([r["sigma"] for r in records], dtype=float)
        if (not np.isfinite(values).all() or not np.isfinite(sigma).all()
                or np.any(sigma <= 0) or not np.isfinite(basis).all()
                or np.any(basis <= 0)):
            raise ValueError("Calibration requires finite values and positive sigma and controls.")
        # The prediction is linear in sqrt(D), with known instrument errors.
        weighted_basis = basis / sigma
        amplitude = np.dot(weighted_basis, values / sigma) / np.dot(
            weighted_basis, weighted_basis
        )
        amplitude = np.clip(amplitude, np.sqrt(0.8), np.sqrt(1.2))
        self.diffusivity = float(amplitude ** 2)
        return self

    def predict(self, experiments):
        if self.diffusivity is None:
            raise RuntimeError("Fit the model before predicting.")
        return predict_at(experiments, self.diffusivity)
