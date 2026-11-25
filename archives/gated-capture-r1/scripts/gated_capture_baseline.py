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
        from scipy.optimize import minimize_scalar
        experiments = [r['input'] for r in records]
        values = np.asarray([r['value'] for r in records])
        sigma = np.asarray([r['sigma'] for r in records])
        result = minimize_scalar(lambda D: np.sum(((predict_at(experiments,D)-values)/sigma)**2),
                                 bounds=(.7,1.3),method='bounded',options={'xatol':1e-12})
        self.diffusivity = float(result.x)
        return self

    def predict(self, experiments):
        return predict_at(experiments, self.diffusivity)
