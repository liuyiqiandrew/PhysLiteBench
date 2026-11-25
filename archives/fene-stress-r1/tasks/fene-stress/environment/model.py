"""Steady planar polymer stress."""
import numpy as np
from scipy.optimize import brentq


def conformation(rate, temperature, drag, max_length):
    wi = drag*rate/4
    if max_length is None:
        return temperature*np.diag([1/(1-2*wi), 1/(1+2*wi)]), 1.0
    b = max_length**2/temperature
    lower = max(1.0, 2*abs(wi))+1e-12
    factor = brentq(
        lambda f: 1-1/f-(1/(f-2*wi)+1/(f+2*wi))/b,
        lower, 100+2*abs(wi), xtol=1e-13,
    )
    tensor = temperature*np.diag([1/(factor-2*wi), 1/(factor+2*wi)])
    return tensor, factor


def predict_at(experiments, drag):
    values = []
    for e in experiments:
        tensor, factor = conformation(e['rate'], e['temperature'], drag, e['max_length'])
        values.append(factor*(tensor[0, 0]-tensor[1, 1]))
    return np.array(values)


class Model:
    def __init__(self):
        self.drag = None

    def fit(self, records):
        raise NotImplementedError('Fit drag from the calibration records.')

    def predict(self, experiments):
        if self.drag is None:
            raise RuntimeError('Call fit before predict.')
        return predict_at(experiments, self.drag)
