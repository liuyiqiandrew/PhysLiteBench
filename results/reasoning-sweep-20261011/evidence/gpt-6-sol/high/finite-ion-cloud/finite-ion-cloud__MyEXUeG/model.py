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
                                   e['ion_count'],e['radius_ratio']) for e in experiments],
                      dtype=float)


class Model:
    def __init__(self):
        self.coupling = None

    def fit(self,records):
        # The absolute disk size cancels from the dimensionless pressure.  Pool
        # repeated preparations using their known Gaussian precisions.
        grouped = {}
        for record in records:
            inp = record['input']
            key = (inp['temperature_multiplier'], inp['ion_count'],
                   inp['radius_ratio'])
            precision = 1.0 / record['sigma']**2
            total, weighted = grouped.get(key, (0.0, 0.0))
            grouped[key] = (total + precision,
                            weighted + precision * record['value'])

        def loss(g):
            return sum(precision * (coefficient(u*g, n, ratio)
                                    - weighted/precision)**2
                       for (u,n,ratio),(precision,weighted) in grouped.items())

        result = minimize_scalar(loss, bounds=(0.7,1.4), method='bounded',
                                 options={'xatol':1e-12})
        self.coupling = float(result.x)
        return self

    def predict(self,experiments):
        return predict_at(experiments,self.coupling)


@lru_cache(maxsize=None)
def nodes(order):
    x,w = roots_legendre(order)
    return (x+1)/2,w/2


def three_ion_coefficient(gamma,ratio,order=32,modes=16):
    length = np.log(ratio)
    x,w = nodes(order)
    d1 = np.broadcast_to((length*x)[:,None],(order,order)).ravel()
    d2 = ((length-length*x)[:,None]*x[None,:]).ravel()
    weights = (length*(length-length*x)[:,None]*w[:,None]*w[None,:]).ravel()
    # For three ions the relative pair angles are constrained: the angle
    # between ions 1 and 3 is the sum of the other two.  If c_m(q) is the
    # Fourier coefficient of |1-q exp(i theta)|^gamma, the exact angular
    # average is sum_m c_m(q12)c_m(q23)c_m(q13).
    p = -gamma/2
    qs = [np.exp(-gap) for gap in (d1,d2,d1+d2)]
    zs = [q*q for q in qs]
    angular = np.prod([hyp2f1(p,p,1,z) for z in zs],axis=0)
    prefactors = [np.ones_like(q) for q in qs]
    for m in range(1,modes+1):
        for prefactor,q in zip(prefactors,qs):
            prefactor *= q*(p+m-1)/m
        angular += 2*np.prod([prefactor*hyp2f1(p,p+m,m+1,z)
                              for prefactor,z in zip(prefactors,zs)],axis=0)
    span = length-d1-d2
    slope = 6-6*gamma
    weight = weights*angular*np.exp((4-3*gamma)*d1+(2-gamma)*d2)
    partition = np.dot(weight,span*exprel(slope*span))
    boundary = np.dot(weight,np.exp(slope*span))
    return -np.expm1(-2*length)*boundary/(6*partition)
