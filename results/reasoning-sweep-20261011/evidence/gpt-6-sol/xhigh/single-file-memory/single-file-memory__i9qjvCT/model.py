import numpy as np


def covariance(diffusivity, density, time_a, time_b):
    """Long-time tagged-particle covariance for a fixed initial lattice."""
    return (np.sqrt(diffusivity / np.pi) / density
            * (np.sqrt(time_a + time_b) - np.sqrt(abs(time_a - time_b))))


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
        records = list(records)
        if not records:
            raise ValueError("At least one calibration record is required")

        # The covariance is linear in sqrt(D), so its Gaussian maximum
        # likelihood estimate is a weighted least-squares slope.
        response = predict_at((r["input"] for r in records), 1.0)
        values = np.asarray([r["value"] for r in records], dtype=float)
        sigmas = np.asarray([r["sigma"] for r in records], dtype=float)
        weights = 1.0 / sigmas**2
        amplitude = np.sum(weights * response * values) / np.sum(weights * response**2)
        self.diffusivity = float(np.clip(amplitude, np.sqrt(0.8), np.sqrt(1.2))**2)
        return self

    def predict(self, experiments):
        if self.diffusivity is None:
            raise ValueError("Fit the model before predicting")
        return predict_at(experiments, self.diffusivity)
