import numpy as np


def covariance(diffusivity, density, time_a, time_b):
    # The same fixed lattice is prepared for every shot, so initial-density
    # fluctuations do not contribute to the long-time covariance.
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
            raise ValueError("At least one calibration record is required.")

        # Covariance is linear in sqrt(D), so its Gaussian weighted least
        # squares estimate has a closed form.
        basis = predict_at((record["input"] for record in records), 1.0)
        values = np.asarray([record["value"] for record in records], dtype=float)
        sigmas = np.asarray([record["sigma"] for record in records], dtype=float)
        weights = 1.0 / sigmas**2
        sqrt_diffusivity = np.sum(weights * basis * values) / np.sum(weights * basis**2)
        sqrt_diffusivity = np.clip(sqrt_diffusivity, np.sqrt(0.8), np.sqrt(1.2))
        self.diffusivity = float(sqrt_diffusivity**2)
        return self

    def predict(self, experiments):
        return predict_at(experiments, self.diffusivity)
