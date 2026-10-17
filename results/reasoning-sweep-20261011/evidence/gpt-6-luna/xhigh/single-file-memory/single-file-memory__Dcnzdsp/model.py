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
        # For fixed density and times, the limiting covariance is proportional
        # to sqrt(D).  Fit that linear scale parameter with the known Gaussian
        # measurement uncertainties, then convert it back to D.
        coefficients = []
        values = []
        uncertainties = []
        for record in records:
            experiment = record["input"]
            coefficients.append(covariance(
                1.0,
                float(experiment["density"]),
                float(experiment["time_a"]),
                float(experiment["time_b"]),
            ))
            values.append(float(record["value"]))
            uncertainties.append(float(record["sigma"]))

        coefficients = np.asarray(coefficients, dtype=float)
        values = np.asarray(values, dtype=float)
        uncertainties = np.asarray(uncertainties, dtype=float)
        if coefficients.size == 0:
            raise ValueError("At least one calibration record is required.")
        if (not np.isfinite(coefficients).all()
                or not np.isfinite(values).all()
                or not np.isfinite(uncertainties).all()
                or np.any(uncertainties <= 0)):
            raise ValueError("Calibration values and positive uncertainties must be finite.")

        weights = 1.0 / uncertainties**2
        sqrt_diffusivity = np.sum(weights * coefficients * values) / np.sum(
            weights * coefficients**2
        )
        sqrt_diffusivity = np.clip(sqrt_diffusivity, np.sqrt(0.8), np.sqrt(1.2))
        self.diffusivity = float(sqrt_diffusivity**2)
        return self

    def predict(self, experiments):
        if self.diffusivity is None:
            raise ValueError("Call fit(records) before predict(experiments).")
        return predict_at(experiments, self.diffusivity)
