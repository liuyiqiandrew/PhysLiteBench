import numpy as np
from scipy.optimize import brentq


def response(e):
    rho, mean = e['density'], e['length']
    statistic = e['statistic']
    if statistic == 'mean':
        return 0.
    if statistic == 'variance':
        return rho*(1-rho*mean)

    # Removing the rods' occupied lengths maps the gas to point particles
    # with density c.  For a nonzero current, the point-particle boundary
    # shifts by the sum of the lengths in the fixed sequence of crossed rods.
    # The sequence is annealed across preparations, so its length generating
    # function contributes log(cosh(...)) for the two equally likely lengths.
    contrast = e['contrast']
    field = abs(e['counting_field'])
    c = rho/(1-rho*mean)
    width = c*mean*contrast

    def field_at(s):
        argument = width*np.sinh(s)
        log_cosh = np.logaddexp(argument, -argument) - np.log(2.)
        return s + c*mean*np.sinh(s) - log_cosh

    s = brentq(lambda s: field_at(s)-field, 0., field, xtol=1e-14) if field else 0.
    return c*(np.cosh(s)-1.)


def predict_at(experiments, speed):
    return speed*np.asarray([response(e) for e in experiments])


class Model:
    def __init__(self):
        self.speed = None

    def fit(self, records):
        basis = np.asarray([response(record['input']) for record in records], dtype=float)
        values = np.asarray([record['value'] for record in records], dtype=float)
        sigma = np.asarray([record['sigma'] for record in records], dtype=float)

        weights = 1. / sigma**2
        denominator = np.sum(weights*basis**2)
        if denominator <= 0. or not np.isfinite(denominator):
            raise ValueError('Calibration records do not identify a finite speed.')
        self.speed = float(np.clip(np.sum(weights*basis*values)/denominator, 0.8, 1.3))
        return self

    def predict(self, experiments):
        return predict_at(experiments, self.speed)
