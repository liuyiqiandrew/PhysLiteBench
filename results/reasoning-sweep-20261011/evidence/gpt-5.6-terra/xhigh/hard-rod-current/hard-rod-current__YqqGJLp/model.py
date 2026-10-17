"""Long-time current statistics for the ballistic hard-rod apparatus."""

import numpy as np
from scipy.optimize import brentq


def _log_cosh(x):
    """Compute log(cosh(x)) without overflowing for large finite ``x``."""
    ax = abs(x)
    return ax + np.log1p(np.exp(-2.0 * ax)) - np.log(2.0)


def _free_density(density, length):
    """Density after removing the (mean) excluded rod length."""
    return density / (1.0 - density * length)


def response(experiment):
    """Dimensionless response, i.e. the response at unit rod speed.

    Removing rod lengths maps the system to freely crossing point particles
    of density ``c``.  For a two-length, annealed sequence, the physical
    current field h and the free-particle field s obey, for h >= 0,

        h = s + c*l*sinh(s) - log(cosh(c*l*contrast*sinh(s))).

    The last term is the cumulant generating function of the randomly
    sampled rod lengths.  It vanishes for equal rods.  Symmetry makes the
    SCGF an even function, hence only |counting_field| is needed here.
    """
    density = float(experiment['density'])
    length = float(experiment['length'])
    contrast = float(experiment['contrast'])
    statistic = experiment['statistic']

    if statistic == 'mean':
        return 0.0

    c = _free_density(density, length)
    if statistic == 'variance':
        # c/(1 + c*l)^2, expressed in physical-density variables.
        return density * (1.0 - density * length)

    if statistic != 'scgf':
        raise ValueError("statistic must be 'mean', 'variance', or 'scgf'")

    field = abs(float(experiment['counting_field']))
    if field == 0.0:
        return 0.0

    mean_excluded = c * length
    mixture_scale = mean_excluded * contrast

    def field_from_free(s):
        sinh_s = np.sinh(s)
        return s + mean_excluded * sinh_s - _log_cosh(mixture_scale * sinh_s)

    # field_from_free is strictly increasing on s >= 0.  This expanding
    # bracket is also robust if callers use a field outside the stated range.
    upper = max(1.0, field)
    while field_from_free(upper) < field:
        upper *= 2.0
    s = brentq(lambda value: field_from_free(value) - field, 0.0, upper,
               xtol=1e-13, rtol=1e-14)
    return c * (np.cosh(s) - 1.0)


def predict_at(experiments, speed):
    """Evaluate a batch of experiments at a specified common rod speed."""
    return float(speed) * np.asarray([response(experiment) for experiment in experiments],
                                     dtype=float)


class Model:
    def __init__(self):
        self.speed = None

    def fit(self, records):
        """Fit the common multiplicative speed by Gaussian weighted least squares."""
        unit_response = np.asarray([response(record['input']) for record in records],
                                   dtype=float)
        observed = np.asarray([record['value'] for record in records], dtype=float)
        sigma = np.asarray([record['sigma'] for record in records], dtype=float)
        weights = 1.0 / np.square(sigma)
        denominator = np.sum(weights * np.square(unit_response))
        if not np.isfinite(denominator) or denominator <= 0.0:
            raise ValueError('records must contain at least one informative finite observation')
        self.speed = float(np.sum(weights * unit_response * observed) / denominator)
        return self

    def predict(self, experiments):
        if self.speed is None:
            raise RuntimeError('fit must be called before predict')
        return predict_at(experiments, self.speed)
