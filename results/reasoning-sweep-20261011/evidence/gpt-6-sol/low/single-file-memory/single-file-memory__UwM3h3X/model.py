import numpy as np


def covariance(diffusivity, density, time_a, time_b):
    # A deterministic initial lattice gives the quenched single-file covariance.
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

        basis = predict_at((record["input"] for record in records), 1.0)
        values = np.asarray([record["value"] for record in records], dtype=float)
        sigmas = np.asarray([record["sigma"] for record in records], dtype=float)
        if not (np.isfinite(values).all() and np.isfinite(sigmas).all()
                and np.isfinite(basis).all() and np.all(sigmas > 0)):
            raise ValueError("Calibration records must contain finite values and positive sigmas")

        weighted_basis = basis / sigmas
        weighted_values = values / sigmas
        root_diffusivity = np.dot(weighted_basis, weighted_values) / np.dot(weighted_basis, weighted_basis)
        self.diffusivity = float(np.clip(root_diffusivity, np.sqrt(0.8), np.sqrt(1.2)) ** 2)
        return self

    def predict(self, experiments):
        return predict_at(experiments, self.diffusivity)
