import numpy as np
from scipy.optimize import brentq


def response(e):
    """Current cumulant rate in units in which the rod speed is one.

    A mixture can be mapped to a free point gas by subtracting the rods to
    the left of every centre.  On the Euler (long-time) scale only the mean
    excluded length enters that map.  Thus ``contrast`` affects neither the
    current rate nor its cumulants for the ensemble in the problem.
    """
    rho, mean = float(e['density']), float(e['length'])
    if e['statistic'] == 'mean':
        return 0.0
    if e['statistic'] == 'variance':
        # This is d^2 psi / d lambda^2 at lambda=0.
        return rho * (1.0 - rho * mean)

    # The current distribution is left/right symmetric.  Writing a=|lambda|
    # and solving a = s + c*l*sinh(s) gives its SCGF.
    field = abs(float(e['counting_field']))
    if field == 0.0:
        return 0.0
    c = rho/(1-rho*mean)
    s = brentq(
        lambda s: s + c * mean * np.sinh(s) - field,
        0.0,
        field + 1.0,
        xtol=1e-14,
    )
    return c * 2.0 * np.sinh(s / 2.0) ** 2


def predict_at(experiments, speed):
    return speed*np.asarray([response(e) for e in experiments])


class Model:
    def __init__(self):
        self.speed = None

    def fit(self, records):
        """Fit the single multiplicative speed by weighted least squares."""
        unit_rate = np.asarray([response(record['input']) for record in records], dtype=float)
        values = np.asarray([record['value'] for record in records], dtype=float)
        sigma = np.asarray([record['sigma'] for record in records], dtype=float)

        # All supplied observations are independent Gaussians, so inverse
        # variance weights are the maximum-likelihood fit for their common
        # scale factor.  Zero-response observations contain no speed signal.
        valid = (
            np.isfinite(unit_rate)
            & np.isfinite(values)
            & np.isfinite(sigma)
            & (sigma > 0.0)
        )
        if not np.any(valid & (unit_rate != 0.0)):
            raise ValueError('Calibration records contain no information about speed.')

        x = unit_rate[valid]
        y = values[valid]
        weight = 1.0 / sigma[valid] ** 2
        estimate = np.sum(weight * x * y) / np.sum(weight * x * x)
        # The apparatus specifies this physical range; applying it also makes
        # the estimate well-defined if a noisy calibration lands just outside.
        self.speed = float(np.clip(estimate, 0.8, 1.3))
        return self

    def predict(self, experiments):
        if self.speed is None:
            raise RuntimeError('Call fit(records) before predict(experiments).')
        return predict_at(experiments, self.speed)
