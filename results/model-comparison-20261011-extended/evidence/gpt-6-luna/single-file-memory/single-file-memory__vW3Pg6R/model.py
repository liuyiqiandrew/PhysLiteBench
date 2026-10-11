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
        """Infer the shared diffusivity by weighted least squares.

        For every record the covariance is proportional to ``sqrt(D)``.  The
        Gaussian errors are specified in output units, so fitting the
        coefficient of ``sqrt(D)`` with weights ``1 / sigma**2`` is the
        maximum-likelihood estimate.  The known physical range is applied as
        a bound on the resulting diffusivity.
        """
        numerator = 0.0
        denominator = 0.0

        count = 0
        for record in records:
            experiment = record["input"]
            value = float(record["value"])
            sigma = float(record["sigma"])
            if not np.isfinite(value):
                raise ValueError("Calibration values must be finite.")
            if not np.isfinite(sigma) or sigma <= 0.0:
                raise ValueError("Calibration standard deviations must be finite and positive.")

            unit_prediction = covariance(
                1.0,
                float(experiment["density"]),
                float(experiment["time_a"]),
                float(experiment["time_b"]),
            )
            scaled_prediction = unit_prediction / sigma
            scaled_value = value / sigma
            numerator += scaled_prediction * scaled_value
            denominator += scaled_prediction * scaled_prediction
            count += 1

        if count == 0 or denominator == 0.0:
            raise ValueError("At least one informative calibration record is required.")

        sqrt_diffusivity = numerator / denominator
        sqrt_diffusivity = float(np.clip(sqrt_diffusivity, np.sqrt(0.8), np.sqrt(1.2)))
        self.diffusivity = sqrt_diffusivity * sqrt_diffusivity
        return self

    def predict(self, experiments):
        if self.diffusivity is None:
            raise RuntimeError("Call fit(records) before predict(experiments).")
        return predict_at(experiments, self.diffusivity)
