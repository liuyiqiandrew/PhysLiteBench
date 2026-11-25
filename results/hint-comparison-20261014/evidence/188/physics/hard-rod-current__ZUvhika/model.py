import numpy as np
from scipy.optimize import brentq, minimize_scalar


def _scgf_at_density(field, point_density, crossed_length):
    """Free-particle SCGF with a boundary shift per crossed rod."""
    if field <= 0.0:
        return 0.0

    def equation(s):
        return s + point_density * crossed_length * np.sinh(s) - field

    s = brentq(equation, 0.0, field + 1.0, xtol=1e-13, rtol=1e-14)
    return point_density * (np.cosh(s) - 1.0)


def _free_scgf(field, density, mean_length):
    """SCGF per unit speed for rods all having ``mean_length``."""
    point_density = density / (1.0 - density * mean_length)
    return _scgf_at_density(field, point_density, mean_length)


def _composition_cost(p):
    """Relative entropy of a Bernoulli(p) block against equal proportions."""
    if p <= 0.0 or p >= 1.0:
        return np.log(2.0)
    return p * np.log(2.0 * p) + (1.0 - p) * np.log(2.0 * (1.0 - p))


def _mixed_scgf(field, density, mean_length, contrast):
    """SCGF per unit speed, retaining the crossed-block composition cost."""
    if field <= 0.0 or contrast == 0.0:
        return _free_scgf(field, density, mean_length)

    point_density = density / (1.0 - density * mean_length)
    short = mean_length * (1.0 - contrast)
    long = mean_length * (1.0 + contrast)

    def value(p):
        available_field = field - _composition_cost(p)
        if available_field <= 0.0:
            return 0.0
        crossed_mean = short + (long - short) * p
        return _scgf_at_density(available_field, point_density, crossed_mean)

    # For field < log(2), only the central interval has a nonzero value.
    # Restricting the scalar optimization to that interval avoids the zero
    # plateaux at the two ends.  At larger fields an endpoint can win.
    log_two = np.log(2.0)
    if field < log_two:
        left = brentq(lambda p: _composition_cost(p) - field,
                      0.0, 0.5, xtol=1e-13, rtol=1e-14)
        right = 1.0 - left
    else:
        left, right = 0.0, 1.0

    result = max(value(left), value(right))
    if right - left > 1e-14:
        optimum = minimize_scalar(
            lambda p: -value(p),
            bounds=(left, right),
            method='bounded',
            options={'xatol': 2e-11},
        )
        result = max(result, -optimum.fun)
    return result


def response(e):
    density = float(e['density'])
    mean_length = float(e['length'])
    statistic = e['statistic']

    if statistic == 'mean':
        return 0.0
    if statistic == 'variance':
        # The zero-field curvature depends only on the ordinary mean rod
        # length; composition selection enters the nonzero-field SCGF.
        return density * (1.0 - density * mean_length)

    field = abs(float(e['counting_field']))
    contrast = float(e['contrast'])
    if contrast == 0.0:
        return _free_scgf(field, density, mean_length)
    return _mixed_scgf(field, density, mean_length, contrast)


def predict_at(experiments, speed):
    return speed * np.asarray([response(e) for e in experiments], dtype=float)


class Model:
    def __init__(self):
        self.speed = None

    def fit(self, records):
        # Every response is linear in the common speed.  Weighted least
        # squares gives the one-parameter Gaussian maximum-likelihood fit.
        base = []
        values = []
        weights = []
        for record in records:
            q = response(record['input'])
            sigma = float(record.get('sigma', 1.0))
            if np.isfinite(q) and np.isfinite(sigma) and sigma > 0.0 and q != 0.0:
                base.append(q)
                values.append(float(record['value']))
                weights.append(1.0 / (sigma * sigma))

        if not base:
            raise ValueError('records contain no informative observations')

        base = np.asarray(base)
        values = np.asarray(values)
        weights = np.asarray(weights)
        fitted = np.sum(weights * base * values) / np.sum(weights * base * base)
        self.speed = float(np.clip(fitted, 0.8, 1.3))
        return self

    def predict(self, experiments):
        if self.speed is None:
            raise ValueError('fit must be called before predict')
        return predict_at(experiments, self.speed)
