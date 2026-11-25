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
        """Fit the common ballistic speed by weighted least squares.

        For a fixed preparation the readout is linear in the speed.  The
        calibration uncertainties are independent and supplied as ``sigma``,
        so the maximum-likelihood estimate is the weighted slope through the
        origin.  (The mean readout is identically zero and therefore carries
        no information about the speed.)
        """
        if not records:
            raise ValueError('at least one calibration record is required')

        design = np.asarray([response(record['input']) for record in records],
                            dtype=float)
        values = np.asarray([record['value'] for record in records],
                            dtype=float)
        sigma = np.asarray([record['sigma'] for record in records],
                           dtype=float)

        if (not np.isfinite(design).all() or
                not np.isfinite(values).all() or
                not np.isfinite(sigma).all() or
                np.any(sigma <= 0)):
            raise ValueError('calibration records must contain finite values '
                             'and positive uncertainties')

        weights = 1.0 / sigma**2
        denominator = np.sum(weights * design**2)
        if denominator == 0:
            raise ValueError('calibration records contain no speed-sensitive '
                             'readout')

        self.speed = float(np.sum(weights * design * values) / denominator)
        return self

    def predict(self, experiments):
        return predict_at(experiments, self.speed)
