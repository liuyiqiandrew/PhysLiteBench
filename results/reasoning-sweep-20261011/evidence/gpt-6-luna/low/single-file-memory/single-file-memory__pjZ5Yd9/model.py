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
        """Fit the one shared diffusivity by weighted least squares.

        The theoretical coefficient is proportional to sqrt(diffusivity).  If
        ``g`` is the coefficient at unit diffusivity, the model is ``a*g``
        with ``a = sqrt(diffusivity)``, so the Gaussian weighted least-squares
        estimate of ``a`` has a closed form.
        """
        records = list(records)
        if not records:
            raise ValueError("At least one calibration record is required.")

        unit_coefficients = np.asarray([
            covariance(1.0, float(record["input"]["density"]),
                       float(record["input"]["time_a"]),
                       float(record["input"]["time_b"]))
            for record in records
        ], dtype=float)
        values = np.asarray([float(record["value"]) for record in records], dtype=float)
        sigmas = np.asarray([float(record["sigma"]) for record in records], dtype=float)

        if (not np.all(np.isfinite(unit_coefficients))
                or not np.all(np.isfinite(values))
                or not np.all(np.isfinite(sigmas))
                or np.any(sigmas <= 0)):
            raise ValueError("Calibration records must contain finite values and positive sigmas.")

        weights = 1.0 / np.square(sigmas)
        denominator = np.sum(weights * np.square(unit_coefficients))
        amplitude = np.sum(weights * unit_coefficients * values) / denominator
        self.diffusivity = float(np.clip(amplitude * amplitude, 0.8, 1.2))
        return self

    def predict(self, experiments):
        return predict_at(experiments, self.diffusivity)
