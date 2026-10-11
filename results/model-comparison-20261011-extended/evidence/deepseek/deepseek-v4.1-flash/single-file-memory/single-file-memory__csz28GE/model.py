import numpy as np

_TWO_PI = 2.0 * np.pi


def covariance(diffusivity, density, time_a, time_b):
    prefactor = np.sqrt(diffusivity / _TWO_PI) / density
    return prefactor * (np.sqrt(time_a) + np.sqrt(time_b) - np.sqrt(abs(time_a - time_b)))


def predict_at(experiments, diffusivity):
    return np.asarray([
        covariance(float(diffusivity), float(e["density"]),
                   float(e["time_a"]), float(e["time_b"]))
        for e in experiments
    ], dtype=float)


def _design(inputs):
    """Model response per unit sqrt(D): g = (sqrt(a)+sqrt(b)-sqrt|a-b|)/(density*sqrt(2*pi))."""
    return np.asarray([
        (np.sqrt(float(e["time_a"])) + np.sqrt(float(e["time_b"]))
         - np.sqrt(abs(float(e["time_a"]) - float(e["time_b"]))))
        / (float(e["density"]) * np.sqrt(_TWO_PI))
        for e in inputs
    ], dtype=float)


class Model:
    def __init__(self):
        self.diffusivity = None

    def fit(self, records):
        inputs = [r["input"] for r in records]
        values = np.asarray([float(r["value"]) for r in records], dtype=float)
        sigmas = np.asarray([float(r["sigma"]) for r in records], dtype=float)

        # covariance is linear in sqrt(diffusivity), so weighted least squares
        # on the design g recovers sqrt(diffusivity) in closed form.
        g = _design(inputs)
        weights = 1.0 / sigmas**2
        denom = float(np.sum(weights * g * g))
        numer = float(np.sum(weights * g * values))
        root_d = numer / denom
        self.diffusivity = root_d**2
        return self

    def predict(self, experiments):
        return predict_at(experiments, self.diffusivity)
