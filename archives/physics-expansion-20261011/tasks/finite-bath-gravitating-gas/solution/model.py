from functools import lru_cache
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq


@lru_cache(maxsize=1)
def profile():
    start = 1e-5
    def derivative(x, y):
        return [y[1], np.exp(-y[0])-2*y[1]/x]
    return solve_ivp(derivative, (start, 60.), [start**2/6, start/3],
                     rtol=2e-11, atol=2e-13, dense_output=True, max_step=.1).sol


def milne(alpha):
    psi, gradient = profile()(alpha)
    eta = alpha*gradient
    return alpha**2*np.exp(-psi)/eta, eta


@lru_cache(maxsize=None)
def coefficient(observable, contrast, bath):
    if observable == 'temperature':
        alpha = brentq(lambda x: profile()(x)[0]-np.log(contrast), .01, 9.)
    elif bath == 'infinite':
        alpha = brentq(lambda x: milne(x)[0]-1, 5., 12.)
    else:
        def energy_turning_point(alpha):
            u, v = milne(alpha)
            return 4*u*u+2*u*v+(2*bath-11)*u+3-2*bath
        alpha = brentq(energy_turning_point, 8.99, 40.)
    return 1/milne(alpha)[1]


def predict_at(experiments, gravity):
    return np.asarray([gravity*e['mass']/e['radius']*
                       coefficient(e['observable'], e.get('contrast'), e['bath_capacity'])
                       for e in experiments])


class Model:
    def __init__(self):
        self.gravity = None

    def fit(self, records):
        inputs = [r['input'] for r in records]
        values = np.asarray([r['value'] for r in records])
        sigma = np.asarray([r['sigma'] for r in records])
        design = predict_at(inputs, 1.)/sigma
        self.gravity = float(np.clip(design@(values/sigma)/(design@design), .7, 1.4))
        return self

    def predict(self, experiments):
        return predict_at(experiments, self.gravity)
