import numpy as np


def covariance(diffusivity, density, time_a, time_b):
    """Long-time tagged-particle covariance for a deterministic initial lattice.

    With the initial positions fixed in every shot, only the particles' thermal
    noise contributes.  Integrating the covariance of the net particle current
    across the origin gives this square-root time dependence.
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

        # Covariance is linear in sqrt(D), so weighted least squares gives the
        # Gaussian maximum-likelihood estimate without a numerical optimizer.
        basis = predict_at((record["input"] for record in records), 1.0)
        values = np.asarray([record["value"] for record in records], dtype=float)
        sigmas = np.asarray([record["sigma"] for record in records], dtype=float)
        weights = 1.0 / np.square(sigmas)
        root_diffusivity = np.sum(weights * basis * values) / np.sum(weights * basis**2)
        self.diffusivity = float(np.clip(root_diffusivity, np.sqrt(0.8), np.sqrt(1.2))**2)
        return self

    def predict(self, experiments):
        if self.diffusivity is None:
            raise ValueError("Fit the model before predicting")
        return predict_at(experiments, self.diffusivity)
