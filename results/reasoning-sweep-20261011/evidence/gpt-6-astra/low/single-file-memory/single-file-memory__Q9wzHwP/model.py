import numpy as np


def covariance(diffusivity, density, time_a, time_b):
    """Long-time tagged covariance for the deterministic initial lattice.

    With no initial density fluctuations, the integrated diffusive current
    gives C(t,s) = sqrt(D/pi)/rho * (sqrt(t+s) - sqrt(|t-s|)).
    Dividing the current by rho converts it to the tagged displacement.
    """
    return np.sqrt(diffusivity / np.pi) / density * (
        np.sqrt(time_a + time_b) - np.sqrt(abs(time_a - time_b))
    )


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
        """Fit the common sqrt(D) amplitude using Gaussian instrument errors."""
        records = list(records)
        if not records:
            raise ValueError("At least one calibration record is required.")
        basis = predict_at((r["input"] for r in records), 1.0)
        values = np.asarray([r["value"] for r in records], dtype=float)
        sigma = np.asarray([r["sigma"] for r in records], dtype=float)
        if not (np.isfinite(values).all() and np.isfinite(sigma).all()
                and (sigma > 0).all()):
            raise ValueError("Calibration values must be finite and sigma positive.")
        design = basis / sigma
        amplitude = np.dot(design, values / sigma) / np.dot(design, design)
        amplitude = np.clip(amplitude, np.sqrt(0.8), np.sqrt(1.2))
        self.diffusivity = float(amplitude ** 2)
        return self

    def predict(self, experiments):
        if self.diffusivity is None:
            raise RuntimeError("Fit the model before predicting.")
        return predict_at(experiments, self.diffusivity)
