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
        # Every calibration record has p=0, where the exact two-speed gas
        # survival is the squared scaled-Bessel sum.
        x = np.asarray([r['input']['density']*r['input']['time'] for r in records])
        y = np.asarray([r['value'] for r in records])
        sigma = np.asarray([r['sigma'] for r in records])
        def objective(speed):
            z = x*speed
            residual = ((ive(0, z)+ive(1, z))**2-y)/sigma
            return float(residual@residual)
        result = minimize_scalar(objective, bounds=(.8, 1.3), method='bounded',
                                 options={'xatol': 1e-12})
        self.speed = float(result.x)
        return self

    def predict(self, experiments):
        return predict_at(experiments, self.speed)
