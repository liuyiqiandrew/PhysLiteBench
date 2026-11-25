import numpy as np
from scipy.optimize import brentq


def response(e):
    """Return the dimensionless rod-current response for one preparation.

    The rods occupy a fraction ``rho * length`` of the line, so the
    exponential free-space rate is ``rho / (1 - rho * length)``.  The
    annealed current generating function depends on the length distribution
    only through its mean length; the contrast therefore does not appear in
    this expression.
    """
    density, length = e['density'], e['length']
    statistic = e['statistic']
    if statistic == 'mean':
        return 0.
    if statistic == 'variance':
        return density * (1 - density * length)

    field = abs(e['counting_field'])
    free_rate = density / (1 - density * length)
    auxiliary = brentq(
        lambda x: x + free_rate * length * np.sinh(x) - field,
        0.0,
        field + 1.0,
        xtol=1e-14,
    )
    return free_rate * (np.cosh(auxiliary) - 1.0)


def predict_at(experiments, speed):
    return speed*np.asarray([response(e) for e in experiments])


class Model:
    def __init__(self):
        self.speed = None

    def fit(self, records):
        """Fit the common speed using the known calibration uncertainties."""
        design = []
        values = []
        weights = []
        for record in records:
            sigma = float(record['sigma'])
            if not np.isfinite(sigma) or sigma <= 0:
                raise ValueError('Calibration sigma must be a positive finite number.')
            design.append(response(record['input']))
            values.append(float(record['value']))
            weights.append(1.0 / sigma**2)

        design = np.asarray(design, dtype=float)
        values = np.asarray(values, dtype=float)
        weights = np.asarray(weights, dtype=float)
        informative = np.isfinite(design) & np.isfinite(values) & np.isfinite(weights)
        denominator = np.sum(weights[informative] * design[informative] ** 2)
        if denominator == 0.0:
            raise ValueError('Calibration records contain no speed-sensitive readout.')

        self.speed = float(
            np.sum(weights[informative] * design[informative] * values[informative])
            / denominator
        )
        return self

    def predict(self, experiments):
        return predict_at(experiments, self.speed)
