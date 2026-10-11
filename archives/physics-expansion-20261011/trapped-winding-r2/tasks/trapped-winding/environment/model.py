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
        raise NotImplementedError('Fit mobility from the calibration records.')

    def predict(self, experiments):
        return predict_at(experiments, self.mobility)
