import numpy as np
from scipy.optimize import brentq


def response(e):
    """Response per unit common speed.

    Removing the rod lengths from the available coordinate leaves a point
    gas of density ``rho/(1-rho*length)``.  Consequently the current law
    depends on a two-length preparation only through its mean length; the
    contrast is immaterial in the thermodynamic limit.
    """
    rho, mean = float(e['density']), float(e['length'])
    statistic = e['statistic']

    if statistic == 'mean':
        return 0.0
    if statistic == 'variance':
        return rho * (1.0 - rho * mean)

    field = abs(float(e['counting_field']))
    free_density = rho / (1.0 - rho * mean)

    # The positive solution is unique because the left hand side is strictly
    # increasing.  The stated field domain makes this bracket conservative,
    # including the zero-field endpoint.
    s = brentq(
        lambda x: x + free_density * mean * np.sinh(x) - field,
        0.0,
        field + 1.0,
        xtol=1e-14,
    )
    return free_density * (np.cosh(s) - 1.0)


def predict_at(experiments, speed):
    return speed*np.asarray([response(e) for e in experiments])


class Model:
    def __init__(self):
        self.speed = None

    def fit(self, records):
        # Every readout is linear in the common speed.  Use the supplied
        # Gaussian uncertainties so that records with a small response are
        # not given disproportionate leverage merely because they are small.
        base = np.asarray([response(r['input']) for r in records], dtype=float)
        values = np.asarray([r['value'] for r in records], dtype=float)
        sigma = np.asarray([r['sigma'] for r in records], dtype=float)

        weights = 1.0 / np.square(sigma)
        denominator = np.sum(weights * np.square(base))
        if not np.isfinite(denominator) or denominator <= 0.0:
            raise ValueError('Calibration records contain no informative readout.')

        speed = np.sum(weights * base * values) / denominator
        # The apparatus specifies the admissible speed interval.  Clipping
        # also keeps a noisy calibration fit physically meaningful.
        self.speed = float(np.clip(speed, 0.8, 1.3))
        return self

    def predict(self, experiments):
        return predict_at(experiments, self.speed)
