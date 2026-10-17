import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize_scalar
from scipy.special import exprel, hyp2f1, roots_legendre
from functools import lru_cache


@lru_cache(maxsize=None)
def coefficient(gamma,count,ratio):
    if count == 3:
        return three_ion_coefficient(gamma,ratio)
    length = np.log(ratio)
    if count == 1:
        slope = 2-gamma
        return -np.expm1(-2*length)*np.exp(slope*length)/(2*length*exprel(slope*length))
    a,b = 4-3*gamma,2-gamma
    def angular(gap):
        return hyp2f1(-gamma/2,-gamma/2,1,np.exp(-2*gap))
    partition = quad(lambda d: angular(d)*np.exp(b*d)*(length-d)*exprel(a*(length-d)),
                     0,length,epsabs=1e-11,epsrel=1e-11)[0]
    boundary = quad(lambda d: angular(d)*np.exp(b*d+a*(length-d)),
                    0,length,epsabs=1e-11,epsrel=1e-11)[0]
    return -np.expm1(-2*length)*boundary/(4*partition)


def predict_at(experiments,coupling):
    return np.asarray([coefficient(e['temperature_multiplier']*coupling,
                                  e['ion_count'],e['radius_ratio']) for e in experiments])


class Model:
    def __init__(self):
        self.coupling = None

    def fit(self,records):
        """Fit the shared reference coupling using the stated measurement errors."""
        records = list(records)
        if not records:
            raise ValueError('At least one calibration record is required.')

        observed = np.asarray([record['value'] for record in records], dtype=float)
        sigma = np.asarray([record['sigma'] for record in records], dtype=float)
        if (not np.all(np.isfinite(observed)) or not np.all(np.isfinite(sigma))
                or np.any(sigma <= 0)):
            raise ValueError('Calibration values must be finite and sigmas positive.')
        experiments = [record['input'] for record in records]
        multipliers = np.asarray(
            [experiment['temperature_multiplier'] for experiment in experiments],
            dtype=float)
        if not np.all(np.isfinite(multipliers)) or np.any(multipliers <= 0):
            raise ValueError('Temperature multipliers must be finite and positive.')

        def chi_squared(coupling):
            predicted = predict_at(experiments, coupling)
            residual = (predicted - observed) / sigma
            return float(np.dot(residual, residual))

        result = minimize_scalar(chi_squared, bounds=(0.7, 1.4), method='bounded',
                                 options={'xatol': 1e-12})
        # Include the endpoints explicitly, since a bounded scalar minimizer
        # does not evaluate either one as a candidate minimum.
        candidates = [(result.fun, result.x), (chi_squared(0.7), 0.7),
                      (chi_squared(1.4), 1.4)]
        self.coupling = float(min(candidates, key=lambda item: item[0])[1])
        return self

    def predict(self,experiments):
        return predict_at(experiments,self.coupling)


@lru_cache(maxsize=None)
def nodes(order):
    x,w = roots_legendre(order)
    return (x+1)/2,w/2


def three_ion_coefficient(gamma,ratio,order=32):
    length = np.log(ratio)
    x,w = nodes(order)
    d1 = np.broadcast_to((length*x)[:,None],(order,order)).ravel()
    d2 = ((length-length*x)[:,None]*x[None,:]).ravel()
    weights = (length*(length-length*x)[:,None]*w[:,None]*w[None,:]).ravel()
    angular = np.ones(len(d1))
    for gap in [d1,d2,d1+d2]:
        angular *= hyp2f1(-gamma/2,-gamma/2,1,np.exp(-2*gap))
    span = length-d1-d2
    slope = 6-6*gamma
    weight = weights*angular*np.exp((4-3*gamma)*d1+(2-gamma)*d2)
    partition = np.dot(weight,span*exprel(slope*span))
    boundary = np.dot(weight,np.exp(slope*span))
    return -np.expm1(-2*length)*boundary/(6*partition)
