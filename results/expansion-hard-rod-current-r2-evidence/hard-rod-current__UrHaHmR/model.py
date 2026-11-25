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
        """Fit the one apparatus parameter shared by all observations.

        For a fixed preparation the rod-current response is proportional to
        the common microscopic speed.  Consequently this is a weighted
        straight-line fit through the origin, rather than a separate fit for
        each statistic or preparation.
        """
        responses = np.asarray([response(record['input']) for record in records],
                               dtype=float)
        values = np.asarray([record['value'] for record in records], dtype=float)
        sigmas = np.asarray([record['sigma'] for record in records], dtype=float)

        weights = 1.0 / np.square(sigmas)
        denominator = np.sum(weights * np.square(responses))
        if not np.isfinite(denominator) or denominator <= 0.0:
            raise ValueError('calibration has no usable nonzero response')

        fitted = np.sum(weights * responses * values) / denominator
        # The apparatus specification supplies the physically admissible
        # interval.  Clipping also prevents an unusually noisy calibration
        # set from producing an invalid prediction model.
        self.speed = float(np.clip(fitted, 0.8, 1.3))
        return self

    def predict(self, experiments):
        if self.speed is None:
            raise RuntimeError('fit must be called before predict')
        return predict_at(experiments, self.speed)
