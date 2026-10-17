import numpy as np
from scipy.optimize import brentq


def response(e):
    """Current cumulant rate in units in which the speed is one.

    Exchanging velocities maps the rods to a point gas in its free volume.
    In the thermodynamic current large-deviation limit the random sequence of
    rod lengths contributes only its mean, so ``contrast`` deliberately does
    not occur below.  (Its cumulative fluctuations are subextensive.)
    """
    rho, mean = float(e['density']), float(e['length'])
    if e['statistic'] == 'mean':
        # The velocity marks are symmetric, hence there is no net current.
        return 0.0
    if e['statistic'] == 'variance':
        # This is the second derivative of the SCGF at zero field.
        return rho*(1-rho*mean)

    field = abs(float(e['counting_field']))
    if field == 0.0:
        return 0.0

    # The effective point-particle density in the free volume is c.  The
    # positive solution is unique because the left side is strictly growing.
    c = rho/(1-rho*mean)
    s = brentq(lambda s: s+c*mean*np.sinh(s)-field, 0., field+1, xtol=1e-14)
    return c*2*np.sinh(s/2)**2


def predict_at(experiments, speed):
    return speed*np.asarray([response(e) for e in experiments])


class Model:
    def __init__(self):
        self.speed = None

    def fit(self, records):
        """Fit the single common speed by Gaussian weighted least squares."""
        if not records:
            raise ValueError('at least one calibration record is required')

        unit_response = np.asarray([response(record['input']) for record in records],
                                   dtype=float)
        values = np.asarray([record['value'] for record in records], dtype=float)
        sigmas = np.asarray([record['sigma'] for record in records], dtype=float)
        if (not np.isfinite(unit_response).all() or not np.isfinite(values).all()
                or not np.isfinite(sigmas).all() or np.any(sigmas <= 0.0)):
            raise ValueError('calibration records must have finite values and positive sigmas')

        # All predicted rates are speed times their unit-speed response.  The
        # apparatus bounds are imposed in the fit rather than merely assumed.
        weights = 1.0 / sigmas**2
        denominator = np.sum(weights * unit_response**2)
        if denominator == 0.0:
            raise ValueError('calibration contains no speed-sensitive readout')
        unconstrained = np.sum(weights * unit_response * values) / denominator
        self.speed = float(np.clip(unconstrained, 0.8, 1.3))
        return self

    def predict(self, experiments):
        if self.speed is None:
            raise RuntimeError('fit must be called before predict')
        return predict_at(experiments, self.speed)
