import numpy as np


def covariance(diffusivity, density, time_a, time_b):
    """Limiting tagged-particle covariance for the fixed initial lattice.

    For a deterministic initial configuration, the leading current fluctuations
    come from Brownian motion after time zero. Their covariance is proportional
    to sqrt(t_a + t_b) - sqrt(|t_a - t_b|).
    """
    return (
        np.sqrt(diffusivity / np.pi) / density
        * (np.sqrt(time_a + time_b) - np.sqrt(abs(time_a - time_b)))
    )


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
        # The prediction is linear in sqrt(D): y_i = sqrt(D) * f_i, where
        # f_i is the covariance evaluated at unit diffusivity. The weighted
        # least-squares estimate therefore has a closed form.
        features = []
        values = []
        weights = []
        for record in records:
            experiment = record["input"]
            features.append(
                covariance(
                    1.0,
                    float(experiment["density"]),
                    float(experiment["time_a"]),
                    float(experiment["time_b"]),
                )
            )
            values.append(float(record["value"]))
            sigma = float(record["sigma"])
            if not np.isfinite(sigma) or sigma <= 0:
                raise ValueError("Each calibration sigma must be finite and positive.")
            weights.append(1.0 / (sigma * sigma))

        features = np.asarray(features, dtype=float)
        values = np.asarray(values, dtype=float)
        weights = np.asarray(weights, dtype=float)
        denominator = np.sum(weights * features * features)
        if features.size == 0 or not np.isfinite(denominator) or denominator <= 0:
            raise ValueError("At least one valid calibration record is required.")

        sqrt_diffusivity = np.sum(weights * features * values) / denominator
        sqrt_diffusivity = np.clip(sqrt_diffusivity, np.sqrt(0.8), np.sqrt(1.2))
        self.diffusivity = float(sqrt_diffusivity * sqrt_diffusivity)
        return self

    def predict(self, experiments):
        if self.diffusivity is None:
            raise RuntimeError("Call fit(records) before predict(experiments).")
        return predict_at(experiments, self.diffusivity)
