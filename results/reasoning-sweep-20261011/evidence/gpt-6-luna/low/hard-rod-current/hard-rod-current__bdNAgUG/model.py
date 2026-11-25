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
        # Every readout is proportional to the common ballistic speed.  The
        # variance and generating-function calibrations therefore give a
        # single weighted least-squares estimate of that speed.
        base = np.asarray([response(r['input']) for r in records], dtype=float)
        values = np.asarray([r['value'] for r in records], dtype=float)
        sigma = np.asarray([r['sigma'] for r in records], dtype=float)
        weights = 1.0 / np.square(sigma)
        denom = np.sum(weights * np.square(base))
        if not np.isfinite(denom) or denom <= 0:
            raise ValueError('Calibration records do not identify a speed.')
        self.speed = float(np.sum(weights * base * values) / denom)
        return self

    def predict(self, experiments):
        return predict_at(experiments, self.speed)
