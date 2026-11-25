import numpy as np
from scipy.optimize import brentq


def response(e):
    """Dimensionless current response for a preparation.

    The speed is a common multiplicative scale.  For the annealed binary
    length sequence in the apparatus, the current depends on the length
    distribution through its mean length, so ``contrast`` does not enter the
    limiting response.
    """
    rho, mean = float(e['density']), float(e['length'])
    if e['statistic'] == 'mean':
        return 0.
    if e['statistic'] == 'variance':
        return rho*(1-rho*mean)
    field = abs(float(e['counting_field']))
    c = rho/(1-rho*mean)
    s = brentq(lambda s: s+c*mean*np.sinh(s)-field, 0., field+1, xtol=1e-14)
    return c*2*np.sinh(s/2)**2


def predict_at(experiments, speed):
    return speed*np.asarray([response(e) for e in experiments])


class Model:
    def __init__(self):
        self.speed = None

    def fit(self, records):
        """Fit the common speed by weighted least squares.

        Each readout has the form ``value = speed * response(input)``.  The
        calibration uncertainties are known, so the Gaussian maximum
        likelihood estimate is the weighted one-parameter least-squares
        estimate.  Mean-current records have zero response and consequently
        carry no information about the speed.
        """
        factors = np.asarray([response(r['input']) for r in records], dtype=float)
        values = np.asarray([r['value'] for r in records], dtype=float)
        sigmas = np.asarray([r['sigma'] for r in records], dtype=float)

        usable = np.isfinite(factors) & np.isfinite(values) & np.isfinite(sigmas)
        usable &= (sigmas > 0) & (factors != 0)
        if not np.any(usable):
            raise ValueError('Calibration contains no informative records.')

        f = factors[usable]
        y = values[usable]
        weights = 1.0 / sigmas[usable]**2
        fitted = np.sum(weights*f*y) / np.sum(weights*f*f)
        # The apparatus specifies the physically allowed speed interval.
        self.speed = float(np.clip(fitted, 0.8, 1.3))
        return self

    def predict(self, experiments):
        return predict_at(experiments, self.speed)
