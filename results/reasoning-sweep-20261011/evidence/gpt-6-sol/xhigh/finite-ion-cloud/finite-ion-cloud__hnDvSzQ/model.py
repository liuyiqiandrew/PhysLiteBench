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
        # The absolute disk radius drops out after scaling all positions by a.
        # Combine repeated preparations using their inverse-variance weights.
        groups = {}
        for record in records:
            experiment = record['input']
            key = (experiment['ion_count'],
                   experiment['temperature_multiplier'],
                   experiment['radius_ratio'])
            weight = 1 / record['sigma']**2
            total_weight, weighted_value = groups.get(key, (0.0, 0.0))
            groups[key] = (total_weight + weight,
                           weighted_value + weight * record['value'])

        if not groups:
            raise ValueError('At least one calibration record is required.')

        def loss(g):
            return sum(weight * (coefficient(g * u, count, ratio)
                                 - weighted_value / weight)**2
                       for (count, u, ratio), (weight, weighted_value)
                       in groups.items())

        result = minimize_scalar(loss, bounds=(0.7, 1.4), method='bounded',
                                 options={'xatol': 1e-11})
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
    # Angular averages of the three pair factors are correlated.  If c_k(q)
    # is the k-th Fourier coefficient of |1-q*exp(i*theta)|**gamma, their
    # joint average is c_0(q12)c_0(q23)c_0(q13) plus twice the sum of the
    # products of their matching positive-frequency coefficients.
    qs = [np.exp(-d1), np.exp(-d2), np.exp(-d1-d2)]
    squared = [q*q for q in qs]
    alpha = -gamma/2
    zero = [hyp2f1(alpha,alpha,1,z) for z in squared]
    angular = zero[0]*zero[1]*zero[2]
    factor = 1.0
    for k in range(1,25):
        factor *= (alpha+k-1)/k  # (-gamma/2)_k / k!
        terms = [factor*q**k*hyp2f1(k+alpha,alpha,k+1,z)
                 for q,z in zip(qs,squared)]
        angular += 2*terms[0]*terms[1]*terms[2]
    span = length-d1-d2
    slope = 6-6*gamma
    weight = weights*angular*np.exp((4-3*gamma)*d1+(2-gamma)*d2)
    partition = np.dot(weight,span*exprel(slope*span))
    boundary = np.dot(weight,np.exp(slope*span))
    return -np.expm1(-2*length)*boundary/(6*partition)
