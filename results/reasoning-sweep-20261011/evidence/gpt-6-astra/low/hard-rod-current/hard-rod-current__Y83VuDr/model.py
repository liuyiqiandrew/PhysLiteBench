"""Long-time, annealed center-current statistics for ballistic rods."""

import numpy as np
from scipy.optimize import brentq


def response(e):
    """Return the readout at unit speed.

    Removing the occupied lengths leaves a Poisson gas of density c. For a
    positive current j and crossed-rod mean length a, its current rate function
    is the Legendre transform of c*(cosh(s)-1)-c*a*j*sinh(s).
    The freshly sampled length sequence contributes j times its sample-mean
    rate function. Eliminating a therefore replaces c*length*sinh(s) by
    -log E[exp(-c*rod_length*sinh(s))] in the field equation. This is the
    annealed average; replacing every rod by its mean misses this correction.
    Reflection symmetry supplies the negative-field branch.
    """
    rho, mean = e['density'], e['length']
    statistic = e['statistic']
    if statistic == 'mean':
        return 0.0
    if statistic == 'variance':
        # Length fluctuations first affect terms beyond quadratic in the field.
        return rho * (1.0 - rho * mean)
    if statistic != 'scgf':
        raise ValueError(f'Unknown statistic: {statistic!r}')

    field = abs(e['counting_field'])
    if field == 0.0:
        return 0.0
    c = rho / (1.0 - rho * mean)
    delta = mean * e['contrast']

    def field_equation(s):
        x = c * np.sinh(s)
        # log(cosh(z)) = log1p(2*sinh(z/2)**2), accurate near zero.
        # Use the asymptotic form at large z to avoid overflow.
        z = delta * x
        log_cosh = (np.log1p(2.0 * np.sinh(z / 2.0)**2) if z < 20.0
                    else z + np.log1p(np.exp(-2.0 * z)) - np.log(2.0))
        return s + mean * x - log_cosh - field

    # All lengths are positive, so the equation is strictly increasing and
    # its root lies between zero and |field|.
    s = brentq(field_equation, 0.0, field, xtol=1e-14)
    return c * 2.0 * np.sinh(s / 2.0)**2


def predict_at(experiments, speed):
    return speed * np.asarray([response(e) for e in experiments], dtype=float)


class Model:
    def __init__(self):
        self.speed = None

    def fit(self, records):
        """Fit the shared speed by bounded Gaussian maximum likelihood."""
        records = list(records)
        x = np.asarray([response(r['input']) / r['sigma'] for r in records])
        y = np.asarray([r['value'] / r['sigma'] for r in records])
        information = np.dot(x, x)
        if not np.isfinite(information) or information <= 0.0:
            raise ValueError('Calibration must contain an informative readout.')
        self.speed = float(np.clip(np.dot(x, y) / information, 0.8, 1.3))
        return self

    def predict(self, experiments):
        if self.speed is None:
            raise ValueError('Call fit before predict.')
        return predict_at(experiments, self.speed)
