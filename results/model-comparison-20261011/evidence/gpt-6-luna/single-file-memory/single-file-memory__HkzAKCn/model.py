import numpy as np


def covariance(diffusivity, density, time_a, time_b):
    # In the long-time limit, the tagged-particle displacement is the
    # integrated particle current at the origin divided by the density.
    # Current fluctuations at two times are correlated through the same
    # underlying Brownian trajectories, which gives this two-time kernel.
    return (np.sqrt(diffusivity/np.pi)/density
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
        # The predicted covariance is linear in sqrt(D).  Since all records
        # have known Gaussian errors, its weighted least-squares estimate is
        # the maximum-likelihood estimate for that parameter.
        values = []
        kernels = []
        weights = []
        for record in records:
            experiment = record["input"]
            values.append(float(record["value"]))
            kernels.append(covariance(
                1.0,
                float(experiment["density"]),
                float(experiment["time_a"]),
                float(experiment["time_b"]),
            ))
            sigma = float(record["sigma"])
            weights.append(1.0 / (sigma * sigma))

        values = np.asarray(values, dtype=float)
        kernels = np.asarray(kernels, dtype=float)
        weights = np.asarray(weights, dtype=float)
        sqrt_diffusivity = np.sum(weights * kernels * values) / np.sum(weights * kernels * kernels)
        self.diffusivity = float(np.clip(sqrt_diffusivity, np.sqrt(0.8), np.sqrt(1.2)) ** 2)
        return self

    def predict(self, experiments):
        return predict_at(experiments, self.diffusivity)
