from functools import lru_cache
import numpy as np


def gate_weights(rate01, rate10):
    matrix = np.array([[-rate01, rate10], [1., 1.]])
    return np.linalg.solve(matrix, [0., 1.])


@lru_cache(maxsize=2048)
def capture_rate(diffusivity, radius, reactivity0, reactivity1, rate01, rate10):
    weights = gate_weights(rate01, rate10)
    reactivity = weights @ np.array([reactivity0, reactivity1])
    surface_concentration = diffusivity/(diffusivity + radius*reactivity)
    return float(4*np.pi*radius**2*reactivity*surface_concentration)


def predict_at(experiments, diffusivity):
    return np.asarray([capture_rate(float(diffusivity), *[float(e[k]) for k in
                       ('radius','reactivity0','reactivity1','rate01','rate10')])
                       for e in experiments],dtype=float)


class Model:
    def __init__(self):
        self.diffusivity = None

    def fit(self, records):
        raise NotImplementedError('Fit the common diffusivity from the calibration records.')

    def predict(self, experiments):
        return predict_at(experiments, self.diffusivity)
