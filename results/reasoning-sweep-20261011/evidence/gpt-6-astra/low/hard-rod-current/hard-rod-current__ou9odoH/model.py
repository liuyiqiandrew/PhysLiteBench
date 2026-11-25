"""Annealed long-time center-current readouts for ballistic hard rods."""

import numpy as np
from scipy.optimize import brentq


def response(e):
    """Return a readout at unit speed.

    Removing rod lengths gives a Poisson gas of density c=rho/(1-rho*a).
    A positive current j traverses j*t consecutive rod lengths. If their
    average is b, its rate cost is j*I(b). The free-gas current rate is

        sup_s {j*(s+c*b*sinh(s)) - c*(cosh(s)-1)}.

    Averaging the length sequence before taking the logarithm minimizes
    I(b)+c*b*sinh(s), giving -log E[exp(-c*ell*sinh(s))]. Thus the positive
    field satisfies field=s+c*a*sinh(s)-log cosh(c*a*contrast*sinh(s)).
    Reflection symmetry supplies the negative-field branch. The bounded
    end-rod offsets have no effect on these long-time rates.
    """
    rho, mean = e['density'], e['length']
    statistic = e['statistic']
    if statistic == 'mean':
        return 0.0
    if statistic == 'variance':
        return rho * (1.0 - rho * mean)
    if statistic != 'scgf':
        raise ValueError(f'Unknown statistic: {statistic!r}')

    field = abs(e['counting_field'])
    if field == 0.0:
        return 0.0
    c = rho / (1.0 - rho * mean)
    contrast = e['contrast']

    def equation(s):
        z = c * mean * np.sinh(s)
        x = contrast * z
        # log(cosh(x)), accurate near zero and safe at large x.
        if x < 1.0:
            log_cosh = np.log1p(2.0 * np.sinh(x / 2.0)**2)
        else:
            log_cosh = x + np.log1p(np.exp(-2.0 * x)) - np.log(2.0)
        return s + z - log_cosh - field

    # The derivative is 1+c*a*cosh(s)*(1-contrast*tanh(x)) > 0.
    # Since every rod length is positive, the root lies in [0, field].
    s = brentq(equation, 0.0, field, xtol=1e-14)
    return c * 2.0 * np.sinh(s / 2.0)**2


def predict_at(experiments, speed):
    return speed * np.asarray([response(e) for e in experiments], dtype=float)


class Model:
    def __init__(self):
        self.speed = None

    def fit(self, records):
        """Fit the shared speed by bounded Gaussian maximum likelihood."""
        records = list(records)
        if not records:
            raise ValueError('At least one calibration record is required.')
        unit = np.asarray([response(r['input']) for r in records])
        values = np.asarray([r['value'] for r in records], dtype=float)
        sigma = np.asarray([r['sigma'] for r in records], dtype=float)
        if not np.all(np.isfinite(sigma) & (sigma > 0.0)):
            raise ValueError('Calibration uncertainties must be finite and positive.')
        x, y = unit / sigma, values / sigma
        information = np.dot(x, x)
        if information == 0.0:
            raise ValueError('Calibration records do not identify the speed.')
        self.speed = float(np.clip(np.dot(x, y) / information, 0.8, 1.3))
        return self

    def predict(self, experiments):
        if self.speed is None:
            raise ValueError('Fit the model before predicting.')
        return predict_at(experiments, self.speed)
