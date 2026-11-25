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
        """Infer the shared diffusivity from noisy calibration records.

        The covariance is proportional to ``sqrt(diffusivity)``.  Writing each
        prediction as ``sqrt(D) * basis`` therefore makes the Gaussian
        weighted least-squares estimate a one-parameter linear regression.
        """
        basis = []
        values = []
        weights = []
        for record in records:
            experiment = record["input"]
            density = float(experiment["density"])
            time_a = float(experiment["time_a"])
            time_b = float(experiment["time_b"])
            sigma = float(record["sigma"])
            # covariance(diffusivity, ...) with sqrt(diffusivity) factored out
            basis.append(
                (np.sqrt(time_a) + np.sqrt(time_b)
                 - np.sqrt(abs(time_a - time_b)))
                / (np.sqrt(2.0 * np.pi) * density)
            )
            values.append(float(record["value"]))
            weights.append(1.0 / sigma**2)

        basis = np.asarray(basis, dtype=float)
        values = np.asarray(values, dtype=float)
        weights = np.asarray(weights, dtype=float)
        sqrt_diffusivity = np.sum(weights * basis * values) / np.sum(
            weights * basis**2
        )
        # The stated physical range also protects against pathological noisy
        # or empty calibration data producing an unusable prediction model.
        self.diffusivity = float(np.clip(sqrt_diffusivity**2, 0.8, 1.2))
        return self

    def predict(self, experiments):
        return predict_at(experiments, self.diffusivity)
