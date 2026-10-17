import numpy as np
from scipy.optimize import brentq


def response(e):
    """Unit-speed response of the rod-current model for one input dictionary.

    Returns the readout divided by the common rod speed. The mean current
    vanishes by +/-v symmetry; the variance and SCGF are independent of the
    length contrast (only the mean length enters).
    """
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
    return speed*np.asarray([response(e) for e in experiments], dtype=float)


class Model:
    def __init__(self):
        self.speed = None

    def fit(self, records):
        values = np.asarray([r['value'] for r in records], dtype=float)
        sigma = np.asarray([r['sigma'] for r in records], dtype=float)
        basis = np.asarray([response(r['input']) for r in records], dtype=float)
        weights = 1.0/sigma**2
        denom = np.sum(weights*basis*basis)
        self.speed = float(np.sum(weights*values*basis)/denom)
        return self

    def predict(self, experiments):
        return predict_at(experiments, self.speed)
