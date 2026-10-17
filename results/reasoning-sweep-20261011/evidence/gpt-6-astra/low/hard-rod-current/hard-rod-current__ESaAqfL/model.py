"""Annealed equilibrium current of ballistic rods, in apparatus units."""

import numpy as np
from scipy.optimize import brentq


def response(e):
    """Return the requested long-time readout at unit common speed.

    Removing rod lengths leaves a Poisson gas of density c. For positive
    current j, the compressed observation point moves by minus the sum of
    the lengths of the j*t rods crossed. If their empirical mean is m,
    the current rate function has the saddle expression

        j*s + c*m*j*sinh(s) - c*(cosh(s)-1) + j*D(m),

    where D is the iid binary-length rate function. Averaging lengths
    before taking the logarithm minimizes over m, giving

        field = s - log E_length[exp(-c*length*sinh(s))].

    The SCGF at this saddle is c*(cosh(s)-1). Reflection symmetry gives
    the negative-field branch. Length fluctuations enter beyond quadratic
    order, so the variance uses only the mean length.
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

    def saddle(s):
        z = c * mean * np.sinh(s)
        # log(cosh(x)), accurate also near zero. The stated domain keeps
        # all intermediate hyperbolic functions comfortably finite.
        log_cosh = np.log1p(2.0 * np.sinh(0.5 * contrast * z)**2)
        return s + z - log_cosh - field

    # The saddle equation is strictly increasing: its derivative is
    # 1+c*mean*cosh(s)*(1-contrast*tanh(contrast*z)) > 0.
    s = brentq(saddle, 0.0, field, xtol=5e-324, rtol=1e-14)
    return c * 2.0 * np.sinh(s / 2.0)**2


def predict_at(experiments, speed):
    return speed * np.asarray([response(e) for e in experiments], dtype=float)


class Model:
    def __init__(self):
        self.speed = None

    def fit(self, records):
        records = list(records)
        if not records:
            raise ValueError('At least one informative calibration record is required.')
        sigma = np.asarray([r['sigma'] for r in records], dtype=float)
        if not np.all(np.isfinite(sigma) & (sigma > 0.0)):
            raise ValueError('Calibration standard deviations must be positive and finite.')
        x = np.asarray([response(r['input']) for r in records]) / sigma
        y = np.asarray([r['value'] for r in records], dtype=float) / sigma
        information = np.dot(x, x)
        if not np.isfinite(information) or information <= 0.0 or not np.all(np.isfinite(y)):
            raise ValueError('Calibration must contain finite, speed-informative observations.')
        # Every asymptotic rate scales linearly with speed. This is the
        # bounded Gaussian maximum-likelihood (weighted least-squares) fit.
        self.speed = float(np.clip(np.dot(x, y) / information, 0.8, 1.3))
        return self

    def predict(self, experiments):
        if self.speed is None:
            raise ValueError('Fit the model before predicting.')
        return predict_at(experiments, self.speed)
