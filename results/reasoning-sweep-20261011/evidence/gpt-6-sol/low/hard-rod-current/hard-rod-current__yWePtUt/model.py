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

    # The gap process has density c in free coordinates. For a positive
    # current, the rods crossed by the origin contribute their lengths to
    # the moving free-coordinate boundary. Annealing over their independent
    # binary lengths gives log E[exp(-c * length * sinh(s))].
    def field_at(s):
        z = c*mean*contrast*np.sinh(s)
        log_cosh = np.logaddexp(z, -z) - np.log(2.)
        return s + c*mean*np.sinh(s) - log_cosh

    s = brentq(lambda s: field_at(s)-field, 0., field+1, xtol=1e-14)
    return c*2*np.sinh(s/2)**2


def predict_at(experiments, speed):
    return speed*np.asarray([response(e) for e in experiments])


class Model:
    def __init__(self):
        self.speed = None

    def fit(self, records):
        inputs = [record['input'] for record in records]
        basis = np.asarray([response(e) for e in inputs], dtype=float)
        values = np.asarray([record['value'] for record in records], dtype=float)
        sigma = np.asarray([record['sigma'] for record in records], dtype=float)
        weighted_basis = basis / sigma
        self.speed = float(np.dot(weighted_basis, values / sigma) /
                           np.dot(weighted_basis, weighted_basis))
        self.speed = float(np.clip(self.speed, 0.8, 1.3))
        return self

    def predict(self, experiments):
        return predict_at(experiments, self.speed)
