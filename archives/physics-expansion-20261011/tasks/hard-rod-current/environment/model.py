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
        raise NotImplementedError('Fit the common speed from calibration records.')

    def predict(self, experiments):
        return predict_at(experiments, self.speed)
