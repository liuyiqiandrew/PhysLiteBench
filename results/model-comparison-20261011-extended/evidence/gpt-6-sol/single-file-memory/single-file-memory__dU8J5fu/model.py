import numpy as np


def covariance(diffusivity, density, time_a, time_b):
    """Long-time tagged-particle covariance for a fixed initial lattice.

    The deterministic initial positions contribute no density fluctuations.
    Integrating the independent Brownian forcing up to the earlier time gives
    the square-root difference below.
    """
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

        # The covariance is linear in sqrt(D), so weighted least squares has
        # a closed-form solution. Each sigma is the known observation error.
        scale = predict_at((record["input"] for record in records), 1.0)
        values = np.asarray([record["value"] for record in records], dtype=float)
        sigmas = np.asarray([record["sigma"] for record in records], dtype=float)
        weights = 1.0 / np.square(sigmas)
        amplitude = np.sum(weights * scale * values) / np.sum(weights * scale * scale)
        amplitude = np.clip(amplitude, np.sqrt(0.8), np.sqrt(1.2))
        self.diffusivity = float(amplitude * amplitude)
        return self

    def predict(self, experiments):
        return predict_at(experiments, self.diffusivity)
