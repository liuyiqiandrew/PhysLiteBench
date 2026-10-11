import numpy as np


def covariance(diffusivity, density, time_a, time_b):
    """Long-time tagged-particle covariance for a fixed initial lattice.

    Initial density fluctuations vanish. Integrating the diffusive current
    noise therefore gives sqrt(D/pi)/rho * (sqrt(t+s) - sqrt(|t-s|)).
    Rationalizing the square-root difference avoids cancellation when the
    two times have very different magnitudes.
    """
    time_factor = (2 * np.minimum(time_a, time_b)
                   / (np.sqrt(time_a + time_b)
                      + np.sqrt(np.abs(time_a - time_b))))
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
        """Fit the common diffusivity by constrained Gaussian likelihood.

        Covariance is linear in sqrt(D), so weighted least squares has an
        exact solution; clipping sqrt(D) enforces the known diffusivity range.
        """
        records = list(records)
        if not records:
            raise ValueError("At least one calibration record is required.")
        basis = predict_at([r["input"] for r in records], 1.0)
        values = np.asarray([r["value"] for r in records], dtype=float)
        sigma = np.asarray([r["sigma"] for r in records], dtype=float)
        if (not np.isfinite(values).all()
                or not np.isfinite(sigma).all() or np.any(sigma <= 0)):
            raise ValueError("Calibration values must be finite and sigma positive.")
        weighted_basis = basis / sigma
        amplitude = np.dot(weighted_basis, values / sigma) / np.dot(
            weighted_basis, weighted_basis)
        self.diffusivity = float(np.clip(amplitude, np.sqrt(0.8), np.sqrt(1.2)) ** 2)
        return self

    def predict(self, experiments):
        if self.diffusivity is None:
            raise RuntimeError("Fit the model before predicting.")
        return predict_at(experiments, self.diffusivity)
