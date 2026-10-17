import numpy as np


def covariance(diffusivity, density, time_a, time_b):
    # A fixed initial lattice gives the quenched single-file covariance.
    # In particular, its increments are not stationary after release.
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

        # The prediction is linear in sqrt(D), so Gaussian maximum likelihood
        # reduces to a one-parameter weighted least-squares fit.
        unit_predictions = predict_at((record["input"] for record in records), 1.0)
        values = np.asarray([record["value"] for record in records], dtype=float)
        sigmas = np.asarray([record["sigma"] for record in records], dtype=float)
        if not (np.isfinite(values).all() and np.isfinite(sigmas).all()
                and (sigmas > 0).all()):
            raise ValueError("Calibration values and sigmas must be finite; sigmas must be positive")
        weights = 1.0 / sigmas**2
        amplitude = np.sum(weights * unit_predictions * values) / np.sum(weights * unit_predictions**2)
        self.diffusivity = float(np.clip(amplitude, np.sqrt(0.8), np.sqrt(1.2))**2)
        return self

    def predict(self, experiments):
        return predict_at(experiments, self.diffusivity)
