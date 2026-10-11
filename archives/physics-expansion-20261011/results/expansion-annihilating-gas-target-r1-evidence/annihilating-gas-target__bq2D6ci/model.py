import math

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import ive


# Twelve-term Stehfest inversion is accurate here because the one-sided
# survival transform is algebraic and is only evaluated on the positive real
# axis.  Keeping these at module scope also makes prediction inexpensive.
_STEHFEST_N = 12


def _stehfest_weights(n):
    weights = []
    half = n // 2
    for k in range(1, n + 1):
        total = 0.0
        for j in range((k + 1) // 2, min(k, half) + 1):
            total += (
                j**half
                * math.factorial(2 * j)
                / (
                    math.factorial(half - j)
                    * math.factorial(j)
                    * math.factorial(j - 1)
                    * math.factorial(k - j)
                    * math.factorial(2 * j - k)
                )
            )
        weights.append((-1.0) ** (k + half) * total)
    return np.asarray(weights)


_STEHFEST_WEIGHTS = _stehfest_weights(_STEHFEST_N)


def _quartic_root(x, p):
    """The physical (small-at-large-x) root of the survival quartic."""
    roots = np.roots((p, 0.0, -(2.0 * p + 1.0), -2.0 * (x + 1.0), p - 1.0))
    return float(np.real(roots[np.argmin(np.abs(roots))]))


def _one_sided_survival(scale, p):
    if scale <= 0.0 or p >= 1.0:
        return 1.0
    if p <= 0.0:
        return float(ive(0, scale) + ive(1, scale))

    # For p=0 the quartic degenerates to the familiar Bessel expression.  In
    # the extremely small-p limit it is also preferable to use that expression
    # directly, since the quartic is ill-conditioned even though its physical
    # answer differs only below double precision at the relevant scale.
    if p < 1.0e-8:
        return float(ive(0, scale) + ive(1, scale))

    sample = np.arange(1.0, _STEHFEST_N + 1.0) * np.log(2.0) / scale
    # If T is the Laplace transform of the derivative of q, then (1+T)/x is
    # the transform of q itself, since q(0)=1.
    transformed = np.array(
        [(1.0 + _quartic_root(x, p)) / x for x in sample]
    )
    value = np.log(2.0) / scale * np.dot(_STEHFEST_WEIGHTS, transformed)
    return float(np.clip(value, 0.0, 1.0))


def response(e, speed):
    """Finite-time survival probability of the stationary tagged particle."""
    rho = float(e['density'])
    p = float(e['stationary_fraction'])
    time = float(e['time'])
    scale = rho * float(speed) * time
    one_sided = _one_sided_survival(scale, p)
    return float(one_sided * one_sided)


def predict_at(experiments, speed):
    return np.asarray([response(e, speed) for e in experiments])


class Model:
    def __init__(self):
        self.speed = None

    def fit(self, records):
        if not records:
            raise ValueError('At least one calibration record is required.')

        experiments = [record['input'] for record in records]
        values = np.asarray([record['value'] for record in records], dtype=float)
        sigmas = np.asarray([record.get('sigma', 1.0) for record in records], dtype=float)
        if not np.all(np.isfinite(values)) or not np.all(np.isfinite(sigmas)):
            raise ValueError('Calibration values and uncertainties must be finite.')
        if np.any(sigmas <= 0.0):
            raise ValueError('Calibration uncertainties must be positive.')

        def objective(speed):
            fitted = predict_at(experiments, speed)
            residual = (fitted - values) / sigmas
            return float(np.dot(residual, residual))

        result = minimize_scalar(
            objective,
            bounds=(0.8, 1.3),
            method='bounded',
            options={'xatol': 1.0e-12},
        )
        self.speed = float(result.x)
        return self

    def predict(self, experiments):
        return predict_at(experiments, self.speed)
