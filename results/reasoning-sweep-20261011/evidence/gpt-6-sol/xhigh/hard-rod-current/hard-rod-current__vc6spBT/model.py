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
    ca = c*mean
    contrast = e['contrast']

    # In free-volume coordinates the rods are Poisson points of density c.
    # A current of m crosses a physical length equal to the sum of the m
    # rods' lengths. Averaging this length over the independently ordered
    # two-size mixture gives E[exp(-q*l)] = exp(-q*mean)*cosh(q*mean*contrast).
    # The positive-field saddle therefore obeys
    #   field = s + c*mean*sinh(s) - log(cosh(c*mean*contrast*sinh(s))).
    # Reflection symmetry supplies the response at negative fields.
    def saddle_equation(s):
        x = ca*contrast*np.sinh(s)
        log_cosh = np.logaddexp(x, -x) - np.log(2.)
        return s + ca*np.sinh(s) - log_cosh - field

    s = brentq(saddle_equation, 0., field, xtol=1e-14)
    return c*2*np.sinh(s/2)**2


def predict_at(experiments, speed):
    return speed*np.asarray([response(e) for e in experiments])


class Model:
    def __init__(self):
        self.speed = None

    def fit(self, records):
        basis = np.asarray([response(record['input']) for record in records])
        values = np.asarray([record['value'] for record in records])
        weights = 1/np.square([record['sigma'] for record in records])
        self.speed = float(np.clip(np.dot(weights*basis, values) /
                                   np.dot(weights*basis, basis), 0.8, 1.3))
        return self

    def predict(self, experiments):
        return predict_at(experiments, self.speed)
