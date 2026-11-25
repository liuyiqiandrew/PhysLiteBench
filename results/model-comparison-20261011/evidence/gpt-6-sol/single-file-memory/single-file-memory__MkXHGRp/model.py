import numpy as np


def covariance(diffusivity, density, time_a, time_b):
    """Long-time tagged-particle covariance for a fixed initial lattice."""
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
        # The covariance is linear in sqrt(D), so its Gaussian weighted
        # least-squares estimate has a closed form.
        basis = []
        values = []
        sigmas = []
        for record in records:
            controls = record["input"]
            basis.append(covariance(1.0, float(controls["density"]),
                                    float(controls["time_a"]),
                                    float(controls["time_b"])))
            values.append(float(record["value"]))
            sigmas.append(float(record["sigma"]))

        if not basis:
            raise ValueError("At least one calibration record is required")

        basis = np.asarray(basis)
        values = np.asarray(values)
        sigmas = np.asarray(sigmas)
        weighted_basis = basis / sigmas
        weighted_values = values / sigmas
        sqrt_diffusivity = np.dot(weighted_basis, weighted_values) / np.dot(
            weighted_basis, weighted_basis)
        sqrt_diffusivity = np.clip(sqrt_diffusivity, np.sqrt(0.8), np.sqrt(1.2))
        self.diffusivity = float(sqrt_diffusivity ** 2)
        return self

    def predict(self, experiments):
        if self.diffusivity is None:
            raise RuntimeError("Fit the model before predicting")
        return predict_at(experiments, self.diffusivity)
