"""Steady planar polymer stress."""
import numpy as np
from scipy.linalg import solve_continuous_lyapunov
from scipy.optimize import brentq


def conformation(rate, rotation, temperature, drag, max_length):
    kappa = np.array([[rate, -rotation], [rotation, -rate]])
    if max_length is None:
        a = 2/drag
        den = a*a+rotation*rotation-rate*rate
        total = 2*temperature*(a*a+rotation*rotation)/den
        difference = 2*temperature*rate*a/den
        cross = temperature*rate*rotation/den
        return np.array([[(total+difference)/2, cross],
                         [cross, (total-difference)/2]]), 1.0
    def tensor(factor):
        matrix = 2*factor/drag*np.eye(2)-kappa
        return solve_continuous_lyapunov(matrix, 4*temperature/drag*np.eye(2))
    lower = max(1., drag/2*np.sqrt(max(rate*rate-rotation*rotation, 0)))+1e-11
    factor = brentq(lambda f: 1-1/f-np.trace(tensor(f))/max_length**2,
                   lower, 100+drag*(abs(rate)+abs(rotation)), xtol=1e-13)
    return tensor(factor), factor


def predict_at(experiments, drag):
    result = []
    for e in experiments:
        tensor, factor = conformation(e['rate'], e['rotation'], e['temperature'],
                                      drag, e['max_length'])
        result.append(factor*(tensor[0, 0]-tensor[1, 1]))
    return np.array(result)


class Model:
    def __init__(self):
        self.drag = None

    def fit(self, records):
        raise NotImplementedError('Fit drag from the calibration records.')

    def predict(self, experiments):
        if self.drag is None:
            raise RuntimeError('Call fit before predict.')
        return predict_at(experiments, self.drag)
