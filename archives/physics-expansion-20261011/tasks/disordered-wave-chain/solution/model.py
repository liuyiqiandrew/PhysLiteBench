import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import roots_legendre
from functools import lru_cache


@lru_cache(maxsize=16)
def quadrature(radial=160, angular=192, upper=14.):
    k, kw = roots_legendre(radial)
    k, kw = (k+1)*upper/2, kw*upper/2
    theta, tw = roots_legendre(angular)
    return k, kw, (theta+1)*np.pi/2, tw/2


@lru_cache(maxsize=4096)
def conical_values(transmission, radial=160, angular=192, upper=14.):
    k, kw, theta, tw = quadrature(radial, angular, upper)
    a = np.arccosh(2/transmission-1)
    base = np.cosh(a)+np.sinh(a)*np.cos(theta)
    return (np.cos(k[:, None]*np.log(base))/np.sqrt(base))@tw


def response(e, strength, radial=160, angular=192, upper=14.):
    count = e['count']
    transmission = 1/(1+(strength/e['wave_number'])**2)
    if count <= 2:
        return 1/(1+count*(strength/e['wave_number'])**2)
    k, kw, _, _ = quadrature(radial, angular, upper)
    conical = conical_values(transmission, radial, angular, upper)
    measure = 2*np.pi*k*np.tanh(np.pi*k)/np.cosh(np.pi*k)
    return float(np.dot(kw, measure*conical**count))

def predict_at(experiments, strength):
    return np.asarray([response(e, strength) for e in experiments])


class Model:
    def __init__(self):
        self.strength = None

    def fit(self, records):
        inputs = [r['input'] for r in records]
        values = np.asarray([r['value'] for r in records])
        sigma = np.asarray([r['sigma'] for r in records])
        count = np.asarray([e['count'] for e in inputs])
        wave_number = np.asarray([e['wave_number'] for e in inputs])
        def objective(strength):
            residual = (1/(1+count*(strength/wave_number)**2)-values)/sigma
            return float(residual@residual)
        fitted = minimize_scalar(objective, bounds=(.6,1.), method='bounded',
                                 options={'xatol': 1e-12})
        self.strength = float(fitted.x)
        return self

    def predict(self, experiments):
        return predict_at(experiments, self.strength)
