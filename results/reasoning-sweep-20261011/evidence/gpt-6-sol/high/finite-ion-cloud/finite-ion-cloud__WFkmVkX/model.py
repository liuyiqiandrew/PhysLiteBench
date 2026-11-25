import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize_scalar
from scipy.special import exprel, factorial, hyp2f1, poch, roots_legendre
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
                                   e['ion_count'],e['radius_ratio']) for e in experiments],
                      dtype=float)


class Model:
    def __init__(self):
        self.coupling = None

    def fit(self,records):
        records = list(records)
        # The absolute disk radius drops out after scaling every position by a.
        # Evaluate each distinct preparation once for every trial coupling.
        preparations = np.array([(r['input']['temperature_multiplier'],
                                  r['input']['ion_count'],
                                  r['input']['radius_ratio']) for r in records])
        unique, inverse = np.unique(preparations, axis=0, return_inverse=True)
        measured = np.array([r['value'] for r in records])
        sigma = np.array([r['sigma'] for r in records])

        def weighted_error(g):
            estimates = np.array([coefficient(g*u, int(n), ratio)
                                  for u, n, ratio in unique])
            return np.sum(((estimates[inverse] - measured)/sigma)**2)

        result = minimize_scalar(weighted_error, bounds=(0.7, 1.4),
                                 method='bounded', options={'xatol': 1e-11})
        self.coupling = float(result.x)
        return self

    def predict(self,experiments):
        return predict_at(experiments,self.coupling)


@lru_cache(maxsize=None)
def nodes(order):
    x,w = roots_legendre(order)
    return (x+1)/2,w/2


def three_ion_coefficient(gamma,ratio,order=24):
    length = np.log(ratio)
    x,w = nodes(order)
    d1 = np.broadcast_to((length*x)[:,None],(order,order)).ravel()
    d2 = ((length-length*x)[:,None]*x[None,:]).ravel()
    weights = (length*(length-length*x)[:,None]*w[:,None]*w[None,:]).ravel()
    # The angles of the three pairs share the same two ion angles.  Their
    # joint average is a Fourier contraction, rather than the product of
    # three separate angular averages.  For
    # f_t(theta) = (1 - 2*t*cos(theta) + t*t)^(gamma/2), its Fourier
    # coefficient of order m is the expression below.
    modes = np.arange(17)
    alpha = gamma/2
    t = np.exp(-np.stack((d1,d2,d1+d2)))[...,None]
    c = (poch(-alpha,modes)/factorial(modes) * t**modes
         * hyp2f1(-alpha,modes-alpha,modes+1,t*t))
    products = np.prod(c,axis=0)
    angular = products[:,0] + 2*np.sum(products[:,1:],axis=1)
    span = length-d1-d2
    slope = 6-6*gamma
    weight = weights*angular*np.exp((4-3*gamma)*d1+(2-gamma)*d2)
    partition = np.dot(weight,span*exprel(slope*span))
    boundary = np.dot(weight,np.exp(slope*span))
    return -np.expm1(-2*length)*boundary/(6*partition)
