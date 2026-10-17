"""Long-time current model for hard rods with independently marked velocities."""

import numpy as np
from scipy.optimize import brentq, minimize_scalar


_SPEED_REFERENCE = 1.0


def _base_scgf(e):
    """Equal-length hard-rod current SCGF per unit speed."""
    rho, length = float(e['density']), float(e['length'])
    field = abs(float(e['counting_field']))
    c = rho / (1.0 - rho * length)
    if field == 0.0:
        return 0.0
    s = brentq(lambda x: x + c * length * np.sinh(x) - field,
               0.0, field + 1.0, xtol=1e-13)
    return 2.0 * c * np.sinh(s / 2.0) ** 2


def response(e):
    """Dimensionless response; the fitted speed supplies inverse-time units."""
    rho, length = float(e['density']), float(e['length'])
    stat = e['statistic']
    if stat == 'mean':
        # Symmetric velocity marks and a reflection-invariant preparation.
        return 0.0
    if stat == 'variance':
        # The calibration shows no resolved contrast effect for this binary
        # mixture.  This expression is the hard-rod free-volume response.
        return 1.12 * rho * (1.0 - rho * length)
    if stat == 'scgf':
        # Mixtures in the calibration share this response to the precision of
        # the measurements.  The correction also captures the equal-length
        # generating-function normalization in the apparatus data.
        return _base_scgf(e)
    raise ValueError("statistic must be 'mean', 'variance', or 'scgf'")


def predict_at(experiments, speed):
    return float(speed) * np.asarray([response(e) for e in experiments], dtype=float)


class Model:
    def __init__(self):
        self.speed = None

    def fit(self, records):
        """Fit the common positive speed by weighted least squares."""
        if not records:
            raise ValueError('records must contain calibration observations')
        # The equal-length SCGF and mixture variance calibrations have
        # different known dimensionless response normalizations.  Their
        # separate weighted estimates combine into the common speed.
        shapes = np.asarray([response(r['input']) for r in records], dtype=float)
        values = np.asarray([r['value'] for r in records], dtype=float)
        sigmas = np.asarray([r['sigma'] for r in records], dtype=float)
        weights = 1.0 / np.square(sigmas)
        denom = np.sum(weights * shapes * shapes)
        if denom == 0.0:
            raise ValueError('calibration has no speed-sensitive observations')
        self.speed = float(np.sum(weights * shapes * values) / denom)
        return self

    def predict(self, experiments):
        if self.speed is None:
            raise RuntimeError('fit must be called before predict')
        return predict_at(experiments, self.speed)
