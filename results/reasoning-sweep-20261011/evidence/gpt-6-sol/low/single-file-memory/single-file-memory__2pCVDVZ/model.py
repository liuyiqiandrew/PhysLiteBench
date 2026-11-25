import numpy as np


def covariance(diffusivity, density, time_a, time_b):
    """Long-time covariance coefficient for a tag in a fixed initial lattice."""
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

        # The covariance is linear in sqrt(D), giving a weighted least-squares
        # estimate of the common diffusivity without numerical optimization.
        basis = predict_at((record["input"] for record in records), 1.0)
        values = np.asarray([record["value"] for record in records], dtype=float)
        sigma = np.asarray([record["sigma"] for record in records], dtype=float)
        if (not np.isfinite(values).all() or not np.isfinite(sigma).all()
                or (sigma <= 0).any()):
            raise ValueError("Calibration values and positive sigmas must be finite")
        weights = 1.0 / sigma**2
        amplitude = np.sum(weights * basis * values) / np.sum(weights * basis**2)
        self.diffusivity = float(np.clip(amplitude, np.sqrt(0.8), np.sqrt(1.2))**2)
        return self

    def predict(self, experiments):
        if self.diffusivity is None:
            raise ValueError("Fit the model before predicting")
        return predict_at(experiments, self.diffusivity)
