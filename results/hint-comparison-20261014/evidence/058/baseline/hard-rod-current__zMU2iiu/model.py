import numpy as np
from scipy.optimize import brentq


def response(e):
    """Current SCGF per unit speed for the equilibrium hard-rod gas.

    In free-volume coordinates the rod density is
    ``c = rho / (1 - rho * length)``.  The physical crossing counter is
    related to the free-particle counter by the (implicit) counting field
    below.  Only the mean rod length enters this thermodynamic limit: the
    random ordering of the two rod sizes has the same mean excluded length.
    """
    rho, mean = float(e['density']), float(e['length'])
    if e['statistic'] == 'mean':
        return 0.
    if e['statistic'] == 'variance':
        return rho*(1-rho*mean)

    # Left/right symmetry makes the SCGF even in the counting field.
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
        """Fit the common speed by Gaussian weighted least squares.

        All response functions factor as ``speed * response(input)``, so the
        likelihood has a one-parameter closed-form minimizer.
        """
        records = list(records)
        base = np.asarray([response(record['input']) for record in records],
                          dtype=float)
        values = np.asarray([record['value'] for record in records],
                            dtype=float)
        sigma = np.asarray([record['sigma'] for record in records],
                           dtype=float)
        weights = 1.0 / np.square(sigma)
        denominator = np.sum(weights * np.square(base))
        if not np.isfinite(denominator) or denominator <= 0.0:
            raise ValueError('calibration must contain an informative finite observation')
        speed = np.sum(weights * base * values) / denominator
        if not np.isfinite(speed):
            raise ValueError('could not fit a finite speed')
        # The apparatus specifies the admissible parameter interval, so this
        # is the constrained Gaussian maximum-likelihood estimate.
        self.speed = float(np.clip(speed, 0.8, 1.3))
        return self

    def predict(self, experiments):
        if self.speed is None:
            raise RuntimeError('call fit(records) before predict(experiments)')
        return predict_at(experiments, self.speed)
