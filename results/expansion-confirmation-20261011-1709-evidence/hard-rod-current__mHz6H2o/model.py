import numpy as np
from scipy.optimize import brentq


def response(e):
    """Return the dimensionless current response (with speed one).

    Removing the rod lengths leaves a point-particle gas with density
    ``c = rho / (1 - rho * mean)``.  The current SCGF is conveniently
    parametrized by ``s``; the counting field is

        field = s + c * mean * sinh(s),

    and the SCGF is ``2*c*sinh(s/2)**2``.  In the infinite, annealed
    preparation only the mean rod length survives, so the contrast does
    not enter this response.
    """
    rho, mean = float(e['density']), float(e['length'])
    statistic = e['statistic']
    if statistic == 'mean':
        return 0.0

    free_fraction = 1.0 - rho * mean
    if statistic == 'variance':
        return rho * free_fraction

    field = abs(float(e['counting_field']))
    c = rho / free_fraction
    if field == 0.0:
        return 0.0

    # The upper endpoint is deliberately generous over the documented
    # domain; the left side is monotone for s >= 0.
    s = brentq(
        lambda x: x + c * mean * np.sinh(x) - field,
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
        # All observations share one multiplicative speed.  Weighted least
        # squares is the Gaussian maximum-likelihood estimate.
        design = np.asarray([response(r['input']) for r in records], dtype=float)
        values = np.asarray([r['value'] for r in records], dtype=float)
        sigma = np.asarray([r['sigma'] for r in records], dtype=float)
        weights = 1.0 / sigma**2

        denominator = np.sum(weights * design**2)
        if not np.isfinite(denominator) or denominator == 0.0:
            raise ValueError('Calibration records contain no informative response.')
        estimate = np.sum(weights * design * values) / denominator
        self.speed = float(np.clip(estimate, 0.8, 1.3))
        return self

    def predict(self, experiments):
        return predict_at(experiments, self.speed)
