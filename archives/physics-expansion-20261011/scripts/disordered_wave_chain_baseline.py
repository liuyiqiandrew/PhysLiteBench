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
        inputs = [r['input'] for r in records]
        values = np.asarray([r['value'] for r in records])
        sigma = np.asarray([r['sigma'] for r in records])
        count = np.asarray([e['count'] for e in inputs])
        wave_number = np.asarray([e['wave_number'] for e in inputs])
        def objective(strength):
            residual = (1/(1+count*(strength/wave_number)**2)-values)/sigma
            return float(residual@residual)
        fitted = minimize_scalar(objective, bounds=(.6,1.), method='bounded',
                                 options={'xatol': 1e-12})
        self.strength = float(fitted.x)
        return self

    def predict(self, experiments):
        return predict_at(experiments, self.strength)
