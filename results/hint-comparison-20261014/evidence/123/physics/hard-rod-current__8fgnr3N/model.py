import numpy as np
from scipy.optimize import brentq, minimize_scalar


def _free_scgf(field, density, background_mean, crossed_mean):
    """SCGF for a block with possibly atypical mean rod length."""
    c = density / (1.0 - density * background_mean)
    if field == 0.0:
        return 0.0

    magnitude = abs(float(field))
    hi = magnitude + 1.0
    while hi + c * crossed_mean * np.sinh(hi) < magnitude:
        hi *= 2.0
    s = brentq(
        lambda z: z + c * crossed_mean * np.sinh(z) - field,
        -hi,
        hi,
        xtol=1e-13,
        rtol=1e-14,
    )
    return c * (np.cosh(s) - 1.0)


def _binary_entropy_cost(fraction, probability=0.5):
    """Relative entropy of a Bernoulli fraction from its base probability."""
    if fraction == 0.0 or fraction == 1.0:
        return -np.log(1.0 - probability if fraction == 0.0 else probability)
    return (
        fraction * np.log(fraction / probability)
        + (1.0 - fraction) * np.log((1.0 - fraction) / (1.0 - probability))
    )


def _mixture_scgf(e):
    density = float(e['density'])
    length = float(e['length'])
    contrast = float(e['contrast'])
    field = abs(float(e['counting_field']))

    short = length * (1.0 - contrast)
    long = length * (1.0 + contrast)
    if field == 0.0 or contrast == 0.0:
        return _free_scgf(field, density, length, length)

    # For positive field the optimizer can be restricted to blocks with at
    # least half short rods: they have the larger free crossing SCGF and the
    # entropy cost is symmetric about one half.
    def objective(fraction_short):
        composition_cost = _binary_entropy_cost(fraction_short)
        effective_field = field - composition_cost
        if effective_field <= 0.0:
            return 0.0
        crossed_mean = (
            fraction_short * short + (1.0 - fraction_short) * long
        )
        return _free_scgf(effective_field, density, length, crossed_mean)

    result = minimize_scalar(
        lambda x: -objective(x),
        bounds=(0.5, 1.0),
        method='bounded',
        options={'xatol': 2e-12},
    )
    return max(objective(0.5), objective(1.0), objective(result.x))


def response(e):
    statistic = e['statistic']
    rho = float(e['density'])
    mean = float(e['length'])
    if statistic == 'mean':
        return 0.0
    if statistic == 'variance':
        return rho * (1.0 - rho * mean)
    if statistic == 'scgf':
        return _mixture_scgf(e)
    raise ValueError(f'unknown statistic: {statistic!r}')


def predict_at(experiments, speed):
    return speed*np.asarray([response(e) for e in experiments], dtype=float)


class Model:
    def __init__(self):
        self.speed = None

    def fit(self, records):
        base = np.asarray([response(r['input']) for r in records], dtype=float)
        values = np.asarray([r['value'] for r in records], dtype=float)
        sigma = np.asarray([r['sigma'] for r in records], dtype=float)
        informative = np.isfinite(base) & (base != 0.0) & np.isfinite(values)
        informative &= np.isfinite(sigma) & (sigma > 0.0)
        if not np.any(informative):
            raise ValueError('calibration contains no informative observations')

        weights = 1.0 / sigma[informative]**2
        x = base[informative]
        y = values[informative]
        self.speed = float(np.sum(weights * x * y) / np.sum(weights * x * x))
        return self

    def predict(self, experiments):
        if self.speed is None:
            raise RuntimeError('fit must be called before predict')
        return predict_at(experiments, self.speed)
