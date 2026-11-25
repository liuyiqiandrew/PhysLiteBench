import numpy as np
from scipy.optimize import brentq


def response(e):
    """Dimensionless rod-current response (with the speed factored out).

    In the thermodynamic limit the length sequence enters through its mean
    occupied length only.  Thus the same expression applies to both the
    equal-length calibration preparations and the mixtures.
    """
    rho, length = float(e['density']), float(e['length'])
    if e['statistic'] == 'mean':
        return 0.
    if e['statistic'] == 'variance':
        return rho * (1.0 - rho * length)

    field = abs(float(e['counting_field']))
    if field == 0.0:
        return 0.0

    # c is the density in the free volume.  The implicit equation is
    # monotone, so a bracketed solve remains reliable at the edge of the
    # advertised parameter domain.
    c = rho / (1.0 - rho * length)
    s = brentq(
        lambda x: x + c * length * np.sinh(x) - field,
        0.0,
        field + 1.0,
        xtol=1e-14,
    )
    return c * 2.0 * np.sinh(s / 2.0) ** 2


def predict_at(experiments, speed):
    return speed*np.asarray([response(e) for e in experiments])


class Model:
    def __init__(self):
        self.speed = None

    def fit(self, records):
        # With Gaussian calibration errors, the maximum-likelihood estimate
        # of the common multiplicative speed is weighted least squares.
        design = []
        values = []
        weights = []
        for record in records:
            x = float(response(record['input']))
            sigma = float(record['sigma'])
            value = float(record['value'])
            if not (np.isfinite(x) and np.isfinite(sigma) and sigma > 0.0
                    and np.isfinite(value)):
                raise ValueError('Calibration records must be finite with positive sigma.')
            if x != 0.0:
                design.append(x)
                values.append(value)
                weights.append(1.0 / sigma**2)

        if not design:
            raise ValueError('Calibration contains no speed-sensitive records.')

        x = np.asarray(design)
        y = np.asarray(values)
        w = np.asarray(weights)
        speed = np.sum(w * x * y) / np.sum(w * x * x)

        # The apparatus specifies the physically allowed interval for the
        # shared speed; use that prior when noisy calibration data overshoots.
        self.speed = float(np.clip(speed, 0.8, 1.3))
        return self

    def predict(self, experiments):
        return predict_at(experiments, self.speed)
