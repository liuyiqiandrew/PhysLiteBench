import numpy as np
from scipy.optimize import minimize_scalar


def response(e, strength):
    return 1/(1+e['count']*(strength/e['wave_number'])**2)


def predict_at(experiments, strength):
    return np.asarray([response(e, strength) for e in experiments])


class Model:
    def __init__(self):
        self.strength = None

    def fit(self, records):
        raise NotImplementedError('Fit the common barrier strength from calibration records.')

    def predict(self, experiments):
        return predict_at(experiments, self.strength)
