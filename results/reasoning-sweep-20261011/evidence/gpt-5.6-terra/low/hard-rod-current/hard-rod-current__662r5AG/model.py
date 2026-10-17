import numpy as np
from scipy.optimize import brentq


def response(e):
    """Current cumulant rates in units where the two speeds are +/- 1.

    The free length per unit physical length is ``1-rho*length``.  The
    length marks are annealed in the preparation described in the README;
    consequently their contrast does not enter the extensive current rate
    (only their mean does).
    """
    rho, mean = float(e['density']), float(e['length'])
    if e['statistic'] == 'mean':
        # The velocity distribution is left/right symmetric.
        return 0.0
    if e['statistic'] == 'variance':
        return rho * (1.0 - rho * mean)

    # The current SCGF is even in its counting field.  Eliminating the
    # free-coordinate tilt s gives h = s + c*l*sinh(s), and
    # psi = 2*c*sinh(s/2)^2, with c=rho/(1-rho*l).
    field = abs(float(e['counting_field']))
    if field == 0.0:
        return 0.0
    free_fraction = 1.0 - rho * mean
    c = rho / free_fraction
    s = brentq(
        lambda tilt: tilt + c * mean * np.sinh(tilt) - field,
        0.0,
        field + 1.0,
        xtol=1e-14,
    )
    return c * 2.0 * np.sinh(0.5 * s) ** 2


def predict_at(experiments, speed):
    return speed*np.asarray([response(e) for e in experiments])


class Model:
    def __init__(self):
        self.speed = None

    def fit(self, records):
        """Fit the shared multiplicative speed by Gaussian weighted LS."""
        unit_rates = np.asarray([response(record['input']) for record in records],
                                dtype=float)
        values = np.asarray([record['value'] for record in records], dtype=float)
        sigmas = np.asarray([record['sigma'] for record in records], dtype=float)

        # All supplied calibration responses have nonzero information.  The
        # guard also makes the error useful if a caller supplies only means.
        denominator = np.sum((unit_rates / sigmas) ** 2)
        if not np.isfinite(denominator) or denominator <= 0.0:
            raise ValueError('calibration records contain no informative finite rate')
        numerator = np.sum(unit_rates * values / sigmas ** 2)
        self.speed = float(numerator / denominator)
        return self

    def predict(self, experiments):
        return predict_at(experiments, self.speed)
