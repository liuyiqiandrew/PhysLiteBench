import numpy as np
from scipy.optimize import minimize_scalar


def predict_at(experiments, mobility):
    k = np.asarray([e['stiffness'] for e in experiments])
    temperature = np.asarray([e['temperature'] for e in experiments])
    probe = np.abs([e['probe'] for e in experiments])
    radial = np.asarray([e['radial_probe'] for e in experiments])
    relaxation = mobility*k
    diffusion = mobility*temperature
    shifted = np.sqrt(relaxation**2+4*diffusion*radial)
    radial_decay = 4*diffusion*radial/(shifted+relaxation)
    return probe*relaxation+radial_decay


class Model:
    def __init__(self):
        self.mobility = None

    def fit(self, records):
        inputs = [r['input'] for r in records]
        values = np.asarray([r['value'] for r in records])
        sigma = np.asarray([r['sigma'] for r in records])
        def objective(mobility):
            residual = (predict_at(inputs,mobility)-values)/sigma
            return float(residual@residual)
        fitted = minimize_scalar(objective, bounds=(.7,1.4), method='bounded',
                                 options={'xatol': 1e-12})
        self.mobility = float(fitted.x)
        return self

    def predict(self, experiments):
        return predict_at(experiments, self.mobility)
