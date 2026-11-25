import numpy as np
from scipy.optimize import brentq


def response(e):
    """Unit-speed long-time response for the annealed rod ensemble.

    In free-volume coordinates the centers form a Poisson process with rate
    c.  A current of k rods moves the origin by the sum of their lengths in
    those coordinates.  Averaging the two possible lengths for each crossed
    rod adds log(cosh(c * length * contrast * sinh(s))) to the saddle-point
    equation.  The speed only sets the time scale.
    """
    rho, mean = e['density'], e['length']
    if e['statistic'] == 'mean':
        return 0.
    if e['statistic'] == 'variance':
        return rho*(1-rho*mean)
    field = abs(e['counting_field'])
    if field == 0:
        return 0.
    c = rho/(1-rho*mean)
    contrast = e['contrast']

    def saddle(s):
        cs = c*mean*np.sinh(s)
        z = contrast*cs
        log_cosh = np.logaddexp(z, -z) - np.log(2.)
        return s + cs - log_cosh - field

    s = brentq(saddle, 0., field, xtol=1e-14)
    return 2*c*np.sinh(s/2)**2


def predict_at(experiments, speed):
    return speed*np.asarray([response(e) for e in experiments])


class Model:
    def __init__(self):
        self.speed = None

    def fit(self, records):
        inputs = [record['input'] for record in records]
        unit_responses = np.asarray([response(e) for e in inputs])
        values = np.asarray([record['value'] for record in records], dtype=float)
        sigmas = np.asarray([record['sigma'] for record in records], dtype=float)
        weighted_responses = unit_responses / sigmas
        weighted_values = values / sigmas
        estimate = np.dot(weighted_responses, weighted_values) / np.dot(
            weighted_responses, weighted_responses
        )
        self.speed = float(np.clip(estimate, 0.8, 1.3))
        return self

    def predict(self, experiments):
        return predict_at(experiments, self.speed)
