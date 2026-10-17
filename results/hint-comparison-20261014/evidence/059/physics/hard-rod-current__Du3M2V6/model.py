import numpy as np
from scipy.optimize import brentq
from scipy.optimize import minimize_scalar


def _fixed_length_scgf(field, free_density, rod_length):
    """SCGF coefficient for rods all having ``rod_length``.

    The auxiliary tilt ``s`` is the free-particle tilt.  A finite rod shifts
    the crossing boundary by its length, which gives the implicit relation

        field = s + free_density * rod_length * sinh(s).

    The returned value does not include the common speed.
    """
    field = float(field)
    if field <= 0.0:
        return 0.0
    s = brentq(
        lambda z: z + free_density * rod_length * np.sinh(z) - field,
        0.0,
        field,
        xtol=1e-13,
        rtol=1e-14,
    )
    return free_density * 2.0 * np.sinh(s / 2.0) ** 2


def _binary_composition_cost(long_fraction):
    """D(q || 1/2), with the endpoint values handled without log(0)."""
    q = float(long_fraction)
    if q <= 0.0 or q >= 1.0:
        return np.log(2.0)
    return q * np.log(2.0 * q) + (1.0 - q) * np.log(2.0 * (1.0 - q))


def _mixture_scgf(e):
    """Annealed SCGF coefficient for the two-length mixture.

    If a positive net current crosses a block with long-rod fraction q, the
    block has mean length a(q) and probability cost D(q||1/2) per crossed
    rod.  Thus its contribution is the fixed-length response at the reduced
    field ``|field|-D``.  The optimum is over the composition of the net
    crossed block, not over the ordinary mixture mean.
    """
    rho = float(e['density'])
    mean = float(e['length'])
    contrast = float(e['contrast'])
    field = abs(float(e['counting_field']))
    free_density = rho / (1.0 - rho * mean)

    # This shortcut also avoids an unnecessary numerical optimization for the
    # equal-length calibration records.
    if contrast == 0.0:
        return _fixed_length_scgf(field, free_density, mean)

    short = mean * (1.0 - contrast)
    long = mean * (1.0 + contrast)

    def contribution(q):
        cost = _binary_composition_cost(q)
        effective_field = field - cost
        if effective_field <= 0.0:
            return 0.0
        average_length = short + (long - short) * q
        return _fixed_length_scgf(effective_field, free_density, average_length)

    # For a positive field the shorter rods are preferred, so the optimum is
    # in q <= 1/2.  The symmetric q >= 1/2 branch cannot improve it.
    optimum = minimize_scalar(
        lambda q: -contribution(q),
        bounds=(0.0, 0.5),
        method='bounded',
        options={'xatol': 2e-11},
    )
    return float(max(0.0, -optimum.fun))


def response(e):
    rho, mean = e['density'], e['length']
    statistic = e['statistic']
    if statistic == 'mean':
        return 0.0
    if statistic == 'variance':
        # The quadratic term is insensitive to the composition fluctuation;
        # only the mean excluded length enters the current variance.
        return rho * (1.0 - rho * mean)
    if statistic == 'scgf':
        return _mixture_scgf(e)
    raise ValueError(f'unknown statistic: {statistic!r}')


def predict_at(experiments, speed):
    return speed*np.asarray([response(e) for e in experiments])


class Model:
    def __init__(self):
        self.speed = None

    def fit(self, records):
        if not records:
            raise ValueError('at least one calibration record is required')

        coefficients = np.asarray([response(r['input']) for r in records], dtype=float)
        values = np.asarray([r['value'] for r in records], dtype=float)
        sigmas = np.asarray([r['sigma'] for r in records], dtype=float)
        if not (np.isfinite(coefficients).all() and np.isfinite(values).all()):
            raise ValueError('calibration records must be finite')
        if np.any(sigmas <= 0.0) or not np.isfinite(sigmas).all():
            raise ValueError('calibration uncertainties must be positive and finite')

        weights = 1.0 / sigmas**2
        denominator = np.sum(weights * coefficients**2)
        if denominator == 0.0:
            raise ValueError('calibration records contain no speed-sensitive readout')
        fitted = np.sum(weights * coefficients * values) / denominator
        # The apparatus specifies the admissible interval for the common
        # speed.  Clipping also keeps predictions well-defined for noisy data.
        self.speed = float(np.clip(fitted, 0.8, 1.3))
        return self

    def predict(self, experiments):
        return predict_at(experiments, self.speed)
