import numpy as np
from scipy.optimize import brentq


def response(e):
    rho, mean = e['density'], e['length']
    if e['statistic'] == 'mean':
        return 0.
    if e['statistic'] == 'variance':
        return rho*(1-rho*mean)
    field = abs(e['counting_field'])
    c = rho/(1-rho*mean)
    s = brentq(lambda s: s+c*mean*np.sinh(s)-field, 0., field+1, xtol=1e-14)
    return c*2*np.sinh(s/2)**2


def predict_at(experiments, speed):
    return speed*np.asarray([response(e) for e in experiments])


class Model:
    def __init__(self):
        self.speed = None

    def fit(self, records):
        # Every readout is linear in the common speed.  The calibration
        # uncertainties are known, so use the corresponding weighted
        # least-squares estimate for this one parameter.
        unit_speed = np.asarray([response(record['input']) for record in records], dtype=float)
        values = np.asarray([record['value'] for record in records], dtype=float)
        sigmas = np.asarray([record['sigma'] for record in records], dtype=float)

        weights = 1.0 / np.square(sigmas)
        denominator = np.sum(weights * np.square(unit_speed))
        if not np.isfinite(denominator) or denominator <= 0:
            raise ValueError('Calibration records do not identify a finite speed.')
        self.speed = float(np.sum(weights * unit_speed * values) / denominator)
        return self

    def predict(self, experiments):
        return predict_at(experiments, self.speed)
