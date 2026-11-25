"""Long-time current statistics for the ballistic hard-rod apparatus."""

import numpy as np
from scipy.optimize import brentq


def _log_length_mgf(q, length, contrast):
    """log E[exp(q ell)] for the apparatus' two equally likely lengths."""
    lo = length * (1.0 - contrast)
    hi = length * (1.0 + contrast)
    return np.logaddexp(q * lo, q * hi) - np.log(2.0)


def _field_from_s(s, free_density, length, contrast):
    """Counting field corresponding to the auxiliary free-gas field ``s``.

    Mapping a hard-rod configuration to its free gaps gives a Poisson gas of
    density c.  The random rod lengths shift the physical cut, giving

      lambda = s + 1/2 [log M(c(e**s-1)) - log M(c(e**(-s)-1))],

    where M is the length moment-generating function.  For equal rods this
    is lambda = s + c * length * sinh(s).
    """
    q_plus = free_density * np.expm1(s)
    q_minus = free_density * np.expm1(-s)
    return s + 0.5 * (
        _log_length_mgf(q_plus, length, contrast)
        - _log_length_mgf(q_minus, length, contrast)
    )


def _auxiliary_field(field, free_density, length, contrast):
    """Invert the strictly increasing field map on the input domain."""
    if field == 0.0:
        return 0.0
    bound = abs(field) + 1.0
    while _field_from_s(bound, free_density, length, contrast) < field:
        bound *= 2.0
    while _field_from_s(-bound, free_density, length, contrast) > field:
        bound *= 2.0
    return brentq(
        lambda s: _field_from_s(s, free_density, length, contrast) - field,
        -bound,
        bound,
        xtol=1e-13,
        rtol=1e-14,
    )


def response(e):
    """Statistics per unit speed for one apparatus preparation."""
    rho = float(e['density'])
    length = float(e['length'])
    contrast = float(e['contrast'])
    statistic = e['statistic']
    if statistic == 'mean':
        return 0.0

    free_density = rho / (1.0 - rho * length)
    if statistic == 'variance':
        # Length fluctuations first affect higher even cumulants.
        return free_density / (1.0 + free_density * length) ** 2

    field = float(e['counting_field'])
    s = _auxiliary_field(field, free_density, length, contrast)
    return free_density * (np.cosh(s) - 1.0)


def predict_at(experiments, speed):
    return float(speed) * np.asarray([response(e) for e in experiments], dtype=float)


class Model:
    def __init__(self):
        self.speed = None

    def fit(self, records):
        """Fit the shared speed by the Gaussian maximum-likelihood estimate."""
        unit_response = np.asarray([response(record['input']) for record in records])
        values = np.asarray([record['value'] for record in records], dtype=float)
        sigma = np.asarray([record['sigma'] for record in records], dtype=float)
        weights = 1.0 / np.square(sigma)
        denominator = np.sum(weights * np.square(unit_response))
        if not np.isfinite(denominator) or denominator <= 0.0:
            raise ValueError('Calibration must contain a nonzero response.')
        estimate = np.sum(weights * unit_response * values) / denominator
        self.speed = float(np.clip(estimate, 0.8, 1.3))
        return self

    def predict(self, experiments):
        if self.speed is None:
            raise RuntimeError('Call fit(records) before predict(experiments).')
        return predict_at(experiments, self.speed)
