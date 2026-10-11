import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import ive


def response(e, speed):
    z = (1-e['stationary_fraction'])*e['density']*speed*e['time']
    return float((ive(0, z)+ive(1, z))**2)


def predict_at(experiments, speed):
    return np.asarray([response(e, speed) for e in experiments])


class Model:
    def __init__(self):
        self.speed = None

    def fit(self, records):
        raise NotImplementedError('Fit the common speed from calibration records.')

    def predict(self, experiments):
        return predict_at(experiments, self.speed)
