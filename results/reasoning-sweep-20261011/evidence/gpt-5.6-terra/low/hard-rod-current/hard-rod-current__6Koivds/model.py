import numpy as np
from scipy.optimize import brentq


def response(e):
    """Unit-speed long-time current cumulants.

    In free-length coordinates the density is
    ``c = rho / (1 - rho * mean_length)``.  The saddle point for the
    current generating function is parametrised by ``s`` below.  Although
    individual rod lengths can differ, their conserved random ordering has
    the same thermodynamic current law with the mean excluded length.
    """
    rho, mean = e['density'], e['length']
    if e['statistic'] == 'mean':
        return 0.
    if e['statistic'] == 'variance':
        return rho*(1-rho*mean)
    # Left/right symmetry makes the SCGF even.  Treat zero separately so
    # that brentq is never asked to solve a degenerate bracket.
    field = abs(e['counting_field'])
    if field == 0.:
        return 0.
    c = rho/(1-rho*mean)
    s = brentq(lambda s: s+c*mean*np.sinh(s)-field, 0., field, xtol=1e-14)
    return c*2*np.sinh(s/2)**2


def predict_at(experiments, speed):
    return speed*np.asarray([response(e) for e in experiments])


class Model:
    def __init__(self):
        self.speed = None

    def fit(self, records):
        """Fit the common multiplicative speed by weighted least squares."""
        unit = np.asarray([response(r['input']) for r in records], dtype=float)
        values = np.asarray([r['value'] for r in records], dtype=float)
        sigma = np.asarray([r['sigma'] for r in records], dtype=float)

        # The mean is identically zero, hence such observations carry no
        # information about speed.  This expression remains the Gaussian
        # maximum-likelihood estimate for all other supplied records.
        weight = 1.0 / np.square(sigma)
        denominator = np.sum(weight * np.square(unit))
        if not np.isfinite(denominator) or denominator <= 0.:
            raise ValueError('calibration records contain no speed information')
        self.speed = float(np.sum(weight * unit * values) / denominator)
        return self

    def predict(self, experiments):
        return predict_at(experiments, self.speed)
