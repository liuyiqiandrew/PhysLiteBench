import numpy as np
from scipy.optimize import brentq


def response(e):
    rho, mean = e['density'], e['length']
    if e['statistic'] == 'mean':
        return 0.
    if e['statistic'] == 'variance':
        return rho*(1-rho*mean)

    # Removing the rod lengths leaves a gas of point particles at density c.
    # A current of j rods shifts the cut in this free coordinate by the sum
    # of j rod lengths.  Since the order is redrawn between preparations, its
    # contribution to the annealed generating function is the length MGF.
    c = rho/(1-rho*mean)
    field = abs(e['counting_field'])
    if field == 0:
        return 0.

    contrast = e['contrast']

    # For the equiprobable lengths mean*(1 +/- contrast),
    # log E[exp(z*a)] = z*mean + log(cosh(z*mean*contrast)).
    # The positive root is used for either field sign by current-reversal
    # symmetry.
    def log_length_mgf(z):
        x = z*mean*contrast
        return z*mean + np.logaddexp(x, -x) - np.log(2.)

    def saddle(s):
        z = c*np.sinh(s)
        # A positive current is biased toward shorter traversed rods, hence
        # the negative argument in the annealed length MGF.  For equal rods
        # this reduces to s + c*mean*sinh(s).
        return s - log_length_mgf(-z) - field

    s = brentq(saddle, 0., field, xtol=1e-14)
    return c*2*np.sinh(s/2)**2


def predict_at(experiments, speed):
    return speed*np.asarray([response(e) for e in experiments])


class Model:
    def __init__(self):
        self.speed = None

    def fit(self, records):
        # The response is linear in the common speed, so weighted least
        # squares is the Gaussian maximum-likelihood estimate.
        base = np.asarray([response(r['input']) for r in records], dtype=float)
        values = np.asarray([r['value'] for r in records], dtype=float)
        sigma = np.asarray([r['sigma'] for r in records], dtype=float)
        weights = 1./sigma**2
        denominator = np.sum(weights*base**2)
        if not np.isfinite(denominator) or denominator <= 0:
            raise ValueError('Calibration records do not identify the common speed.')
        speed = np.sum(weights*base*values)/denominator
        self.speed = float(np.clip(speed, 0.8, 1.3))
        return self

    def predict(self, experiments):
        return predict_at(experiments, self.speed)
