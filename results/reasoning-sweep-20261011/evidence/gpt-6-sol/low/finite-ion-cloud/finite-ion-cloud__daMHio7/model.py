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
        def loss(g):
            return sum(((coefficient(r['input']['temperature_multiplier']*g,
                                     r['input']['ion_count'],
                                     r['input']['radius_ratio'])-r['value'])/
                        r['sigma'])**2 for r in records)
        result = minimize_scalar(loss, bounds=(0.7, 1.4), method='bounded',
                                 options={'xatol': 1e-10})
        self.coupling = float(result.x)
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
    # With one angle fixed, angular averaging of the three pair factors
    # selects equal Fourier modes from all three pairs.  Multiplying their
    # separate averages would discard the correlated modes.
    p = gamma/2
    gaps = (d1, d2, d1+d2)
    radii = [np.exp(-gap) for gap in gaps]
    squares = [r*r for r in radii]
    zero = [hyp2f1(-p,-p,1,z) for z in squares]
    angular = zero[0]*zero[1]*zero[2]
    factors = [np.ones_like(r) for r in radii]
    for mode in range(1,33):
        for j in range(3):
            factors[j] *= (mode-1-p)/mode*radii[j]
        modes = [factors[j]*hyp2f1(mode-p,-p,mode+1,squares[j])
                 for j in range(3)]
        angular += 2*modes[0]*modes[1]*modes[2]
    span = length-d1-d2
    slope = 6-6*gamma
    weight = weights*angular*np.exp((4-3*gamma)*d1+(2-gamma)*d2)
    partition = np.dot(weight,span*exprel(slope*span))
    boundary = np.dot(weight,np.exp(slope*span))
    return -np.expm1(-2*length)*boundary/(6*partition)
