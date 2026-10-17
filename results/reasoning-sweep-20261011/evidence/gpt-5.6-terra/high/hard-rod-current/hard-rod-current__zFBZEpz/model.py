import numpy as np
from scipy.optimize import brentq


def response(e):
    """Current cumulants per unit speed for the hard-rod gas.

    Removing the (ordered) rod lengths maps the system to free point
    particles.  In the thermodynamic, annealed preparation only the mean
    excluded length remains in this mapping, so ``contrast`` does not enter
    the answer.  Writing ``c = rho / (1 - rho * length)`` for the free-space
    density, the SCGF is

        c (cosh(s) - 1),     field = s + c * length * sinh(s).

    This also gives the variance directly at zero field.
    """
    rho, mean = float(e['density']), float(e['length'])
    if e['statistic'] == 'mean':
        return 0.0
    if e['statistic'] == 'variance':
        return rho * (1.0 - rho * mean)

    field = abs(e['counting_field'])
    if field == 0.0:
        return 0.0
    c = rho/(1-rho*mean)
    s = brentq(lambda s: s + c * mean * np.sinh(s) - field,
               0.0, field + 1.0, xtol=1e-14)
    return c * 2.0 * np.sinh(s / 2.0)**2


def predict_at(experiments, speed):
    return speed*np.asarray([response(e) for e in experiments])


class Model:
    def __init__(self):
        self.speed = None

    def fit(self, records):
        """Fit the common speed by maximum likelihood.

        Each measurement is an independent Gaussian observation with a known
        standard deviation.  The rod-current response is linear in speed,
        hence the weighted least-squares/MLE estimate has a closed form.
        """
        unit_response = np.asarray([response(record['input'])
                                    for record in records], dtype=float)
        observed = np.asarray([record['value'] for record in records],
                              dtype=float)
        sigma = np.asarray([record['sigma'] for record in records],
                           dtype=float)

        if (unit_response.size == 0 or not np.isfinite(unit_response).all()
                or not np.isfinite(observed).all()
                or not np.isfinite(sigma).all() or np.any(sigma <= 0.0)):
            raise ValueError('records must contain finite observations with positive sigma')

        weights = 1.0 / sigma**2
        denominator = np.sum(weights * unit_response**2)
        if denominator == 0.0:
            raise ValueError('calibration records contain no speed-sensitive response')
        self.speed = float(np.sum(weights * unit_response * observed) /
                           denominator)
        return self

    def predict(self, experiments):
        return predict_at(experiments, self.speed)
