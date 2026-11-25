import numpy as np
from scipy.optimize import brentq


def response(e):
    """Return the speed-independent rod-current response.

    Removing the occupied rod lengths maps the equal-mass collision
    dynamics to freely moving particles.  The resulting pressure/current
    relation uses the free-volume density ``c``.  A polydisperse mixture
    enters this relation through its mean length only.
    """
    rho = float(e['density'])
    mean_length = float(e['length'])
    statistic = e['statistic']

    if statistic == 'mean':
        return 0.0

    free_fraction = 1.0 - rho * mean_length
    c = rho / free_fraction

    if statistic == 'variance':
        # This is the second cumulant of the SCGF below.
        return rho * free_fraction

    field = abs(float(e['counting_field']))
    if field == 0.0:
        return 0.0

    # If s is the auxiliary free-particle counting field, the rod
    # exclusion relation is field = s + c*mean_length*sinh(s).
    def relation(s):
        return s + c * mean_length * np.sinh(s) - field

    s = brentq(relation, 0.0, field + 1.0, xtol=1e-13, rtol=1e-14)
    return float(2.0 * c * np.sinh(s / 2.0) ** 2)


def predict_at(experiments, speed):
    return speed*np.asarray([response(e) for e in experiments])


class Model:
    def __init__(self):
        self.speed = None

    def fit(self, records):
        # Each observation is value = speed * response(input) + Gaussian
        # noise.  The weighted least-squares solution is the maximum
        # likelihood estimate for the supplied known standard deviations.
        numerator = 0.0
        denominator = 0.0
        for record in records:
            basis = response(record['input'])
            sigma = float(record['sigma'])
            weight = 1.0 / (sigma * sigma)
            numerator += weight * basis * float(record['value'])
            denominator += weight * basis * basis

        if denominator == 0.0:
            raise ValueError('Calibration records contain no speed-sensitive observations.')

        # The apparatus specifies the admissible speed interval.
        self.speed = float(np.clip(numerator / denominator, 0.8, 1.3))
        return self

    def predict(self, experiments):
        return predict_at(experiments, self.speed)
