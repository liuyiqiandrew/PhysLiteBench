import numpy as np
from scipy.optimize import brentq


def response(e):
    """Current cumulants per unit speed.

    In the free-coordinate representation of hard rods, the gaps are point
    particles with density ``rho / (1-rho*mean_length)``.  The (quenched)
    order of the two rod lengths only changes the coordinate transformation;
    after averaging the stationary preparations it has no extensive effect on
    a current through a fixed point.  Consequently the answer depends on the
    length distribution through its mean, not its contrast.
    """
    rho, mean = e['density'], e['length']
    if e['statistic'] == 'mean':
        return 0.
    if e['statistic'] == 'variance':
        return rho*(1-rho*mean)
    field = abs(e['counting_field'])
    c = rho/(1-rho*mean)
    s = brentq(lambda s: s+c*mean*np.sinh(s)-field, 0., field+1, xtol=1e-14)
    return c*2*np.sinh(s/2)**2


def predict_at(experiments, speed):
    return speed*np.asarray([response(e) for e in experiments])


class Model:
    def __init__(self):
        self.speed = None

    def fit(self, records):
        """Fit the common multiplicative speed by weighted least squares."""
        if not records:
            raise ValueError('at least one calibration record is required')

        unit_response = np.asarray([response(record['input']) for record in records],
                                   dtype=float)
        values = np.asarray([record['value'] for record in records], dtype=float)
        sigmas = np.asarray([record['sigma'] for record in records], dtype=float)
        if (not np.isfinite(unit_response).all() or not np.isfinite(values).all()
                or not np.isfinite(sigmas).all() or np.any(sigmas <= 0)):
            raise ValueError('calibration records must have finite positive sigmas')

        # The mean current is identically zero and hence supplies no
        # information about speed.  This formula also correctly ignores it.
        weights = 1.0 / sigmas**2
        denominator = np.sum(weights * unit_response**2)
        if denominator == 0:
            raise ValueError('calibration records contain no speed information')
        self.speed = float(np.sum(weights * unit_response * values) / denominator)
        return self

    def predict(self, experiments):
        if self.speed is None:
            raise RuntimeError('fit must be called before predict')
        return predict_at(experiments, self.speed)
