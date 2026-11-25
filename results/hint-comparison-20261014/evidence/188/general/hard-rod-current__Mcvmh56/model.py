import numpy as np
from scipy.optimize import brentq


def response(e):
    """Return the readout per unit speed.

    Subtracting the rod lengths from the available center coordinates maps
    the dynamics to point particles with density

        c = density / (1 - density * length).

    In the thermodynamic limit the random ordering of the two rod lengths
    enters this map only through its mean, ``length``.  If ``s`` is the
    corresponding point-particle counting field, the physical field is

        field = s + c * length * sinh(s),

    and the SCGF is ``c * (cosh(s) - 1)``.  Its second derivative at zero is
    the variance readout.  The symmetry of the velocity marks makes the
    mean current zero.
    """
    rho = float(e['density'])
    mean_length = float(e['length'])
    statistic = e['statistic']

    if statistic == 'mean':
        return 0.0
    if statistic == 'variance':
        return rho * (1.0 - rho * mean_length)
    if statistic != 'scgf':
        raise ValueError(f"unknown statistic: {statistic!r}")

    field = abs(float(e['counting_field']))
    c = rho / (1.0 - rho * mean_length)

    # The root is in [0, field], with s=0 when field=0.  A small extra
    # margin on the upper endpoint also keeps this robust at the boundary.
    if field == 0.0:
        return 0.0
    s = brentq(
        lambda x: x + c * mean_length * np.sinh(x) - field,
        0.0,
        field + 1.0,
        xtol=1e-13,
    )
    return c * (np.expm1(s) + np.expm1(-s)) / 2.0


def predict_at(experiments, speed):
    return speed*np.asarray([response(e) for e in experiments])


class Model:
    def __init__(self):
        self.speed = None

    def fit(self, records):
        # Every useful calibration response is linear in the common speed.
        # Consequently the maximum-likelihood estimate for independent
        # Gaussian observations is a one-parameter weighted least-squares
        # fit.  Mean-current records have zero response and carry no speed
        # information, so they naturally drop out of the normal equations.
        coefficients = []
        values = []
        weights = []
        for record in records:
            coefficient = float(response(record['input']))
            sigma = float(record['sigma'])
            if coefficient != 0.0 and sigma > 0.0:
                coefficients.append(coefficient)
                values.append(float(record['value']))
                weights.append(1.0 / (sigma * sigma))

        if not coefficients:
            raise ValueError('calibration records contain no speed information')

        coefficients = np.asarray(coefficients, dtype=float)
        values = np.asarray(values, dtype=float)
        weights = np.asarray(weights, dtype=float)
        estimate = np.sum(weights * coefficients * values) / np.sum(
            weights * coefficients * coefficients
        )

        # The apparatus specifies the admissible speed interval.  Clipping
        # protects predictions from an unusually noisy calibration set while
        # leaving the interior fit unchanged.
        self.speed = float(np.clip(estimate, 0.8, 1.3))
        return self

    def predict(self, experiments):
        return predict_at(experiments, self.speed)
