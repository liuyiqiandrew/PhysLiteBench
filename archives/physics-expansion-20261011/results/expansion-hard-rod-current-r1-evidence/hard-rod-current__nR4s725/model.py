import numpy as np


def response(e):
    rho, a, p = e['density'], e['length'], e['right_fraction']
    h, d = 1-a*rho, 2*p-1
    if e['statistic'] == 'mean':
        return rho*d
    if e['statistic'] == 'variance':
        return rho*h*(1-a*rho*d*d)
    field = e['counting_field']
    c = rho/h
    plus = c*p*(1-a*rho*d)
    minus = c*(1-p)*(1+a*rho*d)
    return (rho*d*field + plus*(np.expm1(h*field)-h*field)
            + minus*(np.expm1(-h*field)+h*field))


def predict_at(experiments, speed):
    return speed*np.asarray([response(e) for e in experiments])


class Model:
    def __init__(self):
        self.speed = None

    def fit(self, records):
        """Fit the common rod speed from independent Gaussian observations.

        Every response is proportional to the speed, so the Gaussian
        maximum-likelihood estimate is the weighted one-parameter linear
        least-squares estimate.  The physical apparatus restricts the speed
        to the interval given in the README; clipping also keeps the model
        well-defined for an unusually noisy calibration set.
        """
        numerator = 0.0
        denominator = 0.0
        for record in records:
            expected_per_speed = float(response(record['input']))
            sigma = float(record['sigma'])
            weight = 1.0 / (sigma * sigma)
            numerator += weight * expected_per_speed * float(record['value'])
            denominator += weight * expected_per_speed * expected_per_speed

        if denominator <= 0.0 or not np.isfinite(denominator):
            raise ValueError('Calibration records contain no informative responses.')

        estimate = numerator / denominator
        self.speed = float(np.clip(estimate, 0.8, 1.3))
        return self

    def predict(self, experiments):
        return predict_at(experiments, self.speed)
