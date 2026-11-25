import numpy as np
from scipy.optimize import brentq


def response(e):
    rho, mean, contrast = e['density'], e['length'], e['contrast']
    if e['statistic'] == 'mean':
        return 0.
    if e['statistic'] == 'variance':
        return rho*(1-rho*mean)
    field = abs(e['counting_field'])
    c = rho/(1-rho*mean)

    def equation(s):
        b = c*mean*contrast*np.sinh(s)
        return s+c*mean*np.sinh(s)-np.logaddexp(b, -b)+np.log(2.)-field

    s = brentq(equation, 0., field+1, xtol=1e-14)
    return c*2*np.sinh(s/2)**2


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
