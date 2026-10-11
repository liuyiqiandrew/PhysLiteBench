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
        """Infer the common diffusion constant from calibration records.

        The model is linear in ``sqrt(diffusivity)``.  Writing the prediction
        for record ``i`` as ``y_i = a_i * sqrt(D)`` makes the Gaussian
        maximum-likelihood estimate a one-parameter weighted least-squares
        problem.  The diffusivity is then constrained to the range specified
        by the apparatus description.
        """
        features = []
        values = []
        weights = []

        for record in records:
            inputs = record["input"]
            density = float(inputs["density"])
            time_a = float(inputs["time_a"])
            time_b = float(inputs["time_b"])
            value = float(record["value"])
            sigma = float(record.get("sigma", 0.002))

            feature = (
                np.sqrt(time_a) + np.sqrt(time_b)
                - np.sqrt(abs(time_a - time_b))
            ) / (density * np.sqrt(2.0 * np.pi))

            features.append(feature)
            values.append(value)
            weights.append(1.0 / sigma**2)

        features = np.asarray(features, dtype=float)
        values = np.asarray(values, dtype=float)
        weights = np.asarray(weights, dtype=float)

        denominator = np.sum(weights * features**2)
        if features.size == 0 or not np.isfinite(denominator) or denominator <= 0:
            raise ValueError("records must contain at least one valid calibration record")

        sqrt_diffusivity = np.sum(weights * features * values) / denominator
        diffusivity = float(sqrt_diffusivity**2)
        self.diffusivity = float(np.clip(diffusivity, 0.8, 1.2))
        return self

    def predict(self, experiments):
        return predict_at(experiments, self.diffusivity)
