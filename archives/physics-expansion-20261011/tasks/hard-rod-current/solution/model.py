import numpy as np
from scipy.optimize import brentq


def response(e):
    rho, a, p = e['density'], e['length'], e['right_fraction']
    h, d = 1-a*rho, 2*p-1
    if e['statistic'] == 'mean':
        return rho*d
    if e['statistic'] == 'variance':
        return rho*h*(1-a*rho*d*d)
    field = e['counting_field']
    c = rho/h
    s = brentq(lambda z: z+a*c*(p*np.expm1(z)-(1-p)*np.expm1(-z))-field,
               -abs(field)-1, abs(field)+1, xtol=1e-14)
    return c*(p*np.expm1(s)+(1-p)*np.expm1(-s))


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
