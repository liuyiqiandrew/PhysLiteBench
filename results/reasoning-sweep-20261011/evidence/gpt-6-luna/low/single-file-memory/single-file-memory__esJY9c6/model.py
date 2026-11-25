import numpy as np


def covariance(diffusivity, density, time_a, time_b):
    return np.sqrt(diffusivity/(2*np.pi))/density * (np.sqrt(time_a)+np.sqrt(time_b)-np.sqrt(abs(time_a-time_b)))


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
        """Infer the shared diffusivity from Gaussian calibration errors.

        The covariance law is linear in ``sqrt(diffusivity)``.  Fitting in
        that parameter therefore gives the exact weighted least-squares
        solution without a numerical optimizer.
        """
        records = list(records)
        if not records:
            raise ValueError("At least one calibration record is required.")

        # For each record, covariance = coefficient * sqrt(D).
        coefficients = np.asarray([
            covariance(1.0, float(r["input"]["density"]),
                       float(r["input"]["time_a"]),
                       float(r["input"]["time_b"]))
            for r in records
        ], dtype=float)
        values = np.asarray([float(r["value"]) for r in records], dtype=float)
        sigmas = np.asarray([float(r["sigma"]) for r in records], dtype=float)
        if (not np.isfinite(coefficients).all() or
                not np.isfinite(values).all() or
                not np.isfinite(sigmas).all() or np.any(sigmas <= 0)):
            raise ValueError("Calibration values and positive sigmas must be finite.")

        weights = 1.0 / sigmas**2
        root_diffusivity = np.sum(weights * coefficients * values) / np.sum(
            weights * coefficients**2
        )
        # The physical parameter is constrained to the documented range.
        self.diffusivity = float(np.clip(root_diffusivity, np.sqrt(0.8), np.sqrt(1.2))**2)
        return self

    def predict(self, experiments):
        return predict_at(experiments, self.diffusivity)
