import numpy as np
from scipy.optimize import brentq


def response(e):
    rho, mean = e['density'], e['length']
    if e['statistic'] == 'mean':
        return 0.
    if e['statistic'] == 'variance':
        return rho*(1-rho*mean)
    field = abs(e['counting_field'])
    if field == 0:
        return 0.
    c = rho/(1-rho*mean)
    spread = mean*e['contrast']
    # The annealed two-length renewal law has Laplace transform
    # exp(-mean*q)*cosh(spread*q), where q = c*sinh(s).
    def equation(s):
        q = c*np.sinh(s)
        return s + mean*q - np.logaddexp(spread*q, -spread*q) + np.log(2.) - field
    s = brentq(equation, 0., field, xtol=1e-14)
    return c*(np.cosh(s)-1.)


def predict_at(experiments, speed):
    return speed*np.asarray([response(e) for e in experiments])


class Model:
    def __init__(self):
        self.speed = None

    def fit(self, records):
        basis = np.asarray([response(record['input']) for record in records])
        values = np.asarray([record['value'] for record in records], dtype=float)
        weights = 1/np.square([record['sigma'] for record in records])
        self.speed = float(np.clip(np.dot(weights*basis, values) /
                                   np.dot(weights*basis, basis), 0.8, 1.3))
        return self

    def predict(self, experiments):
        return predict_at(experiments, self.speed)
