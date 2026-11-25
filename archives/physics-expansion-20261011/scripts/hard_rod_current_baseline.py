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
        basis = predict_at([r['input'] for r in records], 1.)
        sigma = np.asarray([r['sigma'] for r in records])
        values = np.asarray([r['value'] for r in records])
        self.speed = float(np.sum(basis*values/sigma**2)/np.sum((basis/sigma)**2))
        return self

    def predict(self, experiments):
        return predict_at(experiments, self.speed)
