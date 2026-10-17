import numpy as np


def covariance(diffusivity, density, time_a, time_b):
    """Long-time tagged-particle covariance for a fixed initial lattice.

    The deterministic initial positions contribute no initial density noise.
    Integrating the independent thermal forcing from time zero gives a
    covariance proportional to sqrt(t_a + t_b) - sqrt(|t_a - t_b|).
    """
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
        # Each predicted covariance is g_i * sqrt(D), with g_i evaluated at
        # D = 1. Gaussian errors therefore make this a one-parameter weighted
        # linear least-squares problem.
        scale = []
        values = []
        sigmas = []
        for record in records:
            controls = record["input"]
            scale.append(covariance(1.0, float(controls["density"]),
                                    float(controls["time_a"]),
                                    float(controls["time_b"])))
            values.append(float(record["value"]))
            sigmas.append(float(record["sigma"]))

        if not scale:
            raise ValueError("At least one calibration record is required")

        scale = np.asarray(scale, dtype=float)
        values = np.asarray(values, dtype=float)
        sigmas = np.asarray(sigmas, dtype=float)
        if (not np.isfinite(scale).all() or not np.isfinite(values).all()
                or not np.isfinite(sigmas).all() or np.any(sigmas <= 0)):
            raise ValueError("Calibration records must contain finite values and positive sigmas")

        weighted_scale = scale / sigmas
        weighted_values = values / sigmas
        amplitude = np.dot(weighted_scale, weighted_values) / np.dot(
            weighted_scale, weighted_scale)
        if amplitude <= np.sqrt(0.8):
            self.diffusivity = 0.8
        elif amplitude >= np.sqrt(1.2):
            self.diffusivity = 1.2
        else:
            self.diffusivity = float(amplitude * amplitude)
        return self

    def predict(self, experiments):
        if self.diffusivity is None:
            raise RuntimeError("Fit the model before predicting")
        return predict_at(experiments, self.diffusivity)
