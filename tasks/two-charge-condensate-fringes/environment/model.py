import numpy as np
from scipy.special import zeta

BOUNDS = (.62, 1.34)


def _response(experiments, mass):
    values = []
    k = np.arange(1, 181, dtype=float)
    for e in experiments:
        if e["observable"] != "line_density":
            raise NotImplementedError
        T = e["temperature"]
        gaps = np.array(e["gaps"])
        populations = np.exp(-gaps[:, None] * k / T) @ (k ** -1.5)
        density = (mass * T / (2 * np.pi)) ** 1.5
        values.append(e["density"] - density * (5 * zeta(1.5) + populations.sum()))
    return np.array(values)


class Model:
    def fit(self, records):
        # Set self.mass from the calibration measurements.
        raise NotImplementedError

    def predict(self, experiments):
        return _response(experiments, self.mass)
