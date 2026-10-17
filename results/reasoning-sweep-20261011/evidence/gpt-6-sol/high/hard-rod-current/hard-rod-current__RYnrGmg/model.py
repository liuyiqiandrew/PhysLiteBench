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
    contrast = e['contrast']

    # A signed current j moves the free-coordinate observation point by the
    # total length of the |j| rods it passes. Averaging over their independent
    # two-length sequence gives a log-cosh term in the conjugate field.
    def field_from_s(s):
        z = c*mean*contrast*np.sinh(s)
        return s + c*mean*np.sinh(s) - np.logaddexp(z, -z) + np.log(2.)

    s = brentq(lambda s: field_from_s(s)-field, 0., field, xtol=1e-13)
    return 2*c*np.sinh(s/2.)**2


def predict_at(experiments, speed):
    return speed*np.asarray([response(e) for e in experiments])


class Model:
    def __init__(self):
        self.speed = None

    def fit(self, records):
        base = np.asarray([response(r['input']) for r in records], dtype=float)
        values = np.asarray([r['value'] for r in records], dtype=float)
        sigma = np.asarray([r['sigma'] for r in records], dtype=float)
        weighted = base/sigma
        self.speed = float(np.clip(np.dot(weighted, values/sigma) /
                                   np.dot(weighted, weighted), 0.8, 1.3))
        return self

    def predict(self, experiments):
        return predict_at(experiments, self.speed)
