import numpy as np
from scipy.optimize import brentq


def response(e):
    """Current readout at unit rod speed.

    In free-volume coordinates the rod centers form a Poisson gas of density
    ``c``.  A current of ``j`` rods shifts the free-volume origin by the sum
    of their lengths.  Averaging that sum over the freshly drawn rod order
    gives the log-cosh term in the field equation below.
    """
    rho, mean = e['density'], e['length']
    if e['statistic'] == 'mean':
        return 0.
    if e['statistic'] == 'variance':
        return rho*(1-rho*mean)
    field = abs(e['counting_field'])
    if field == 0.:
        return 0.
    c = rho/(1-rho*mean)
    half_difference = mean*e['contrast']

    def applied_field(s):
        length_bias = c*half_difference*np.sinh(s)
        log_cosh = np.logaddexp(length_bias, -length_bias)-np.log(2.)
        return s+c*mean*np.sinh(s)-log_cosh

    s = brentq(lambda s: applied_field(s)-field, 0., field, xtol=1e-14)
    return 2.*c*np.sinh(s/2.)**2


def predict_at(experiments, speed):
    return speed*np.asarray([response(e) for e in experiments])


class Model:
    def __init__(self):
        self.speed = None

    def fit(self, records):
        unit_speed = np.asarray([response(record['input']) for record in records])
        values = np.asarray([record['value'] for record in records])
        weights = 1./np.asarray([record['sigma'] for record in records])**2
        self.speed = float(np.clip(np.dot(weights*unit_speed, values) /
                                   np.dot(weights*unit_speed, unit_speed),
                                   0.8, 1.3))
        return self

    def predict(self, experiments):
        return predict_at(experiments, self.speed)
