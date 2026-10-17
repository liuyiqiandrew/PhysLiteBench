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
        # Independent Gaussian measurements give a weighted least-squares fit.
        # Repeated preparations have the same theoretical response, so combine
        # their sufficient statistics before doing the one-dimensional search.
        grouped = {}
        for record in records:
            experiment = record['input']
            key = (experiment['ion_count'], experiment['temperature_multiplier'],
                   experiment['radius_ratio'])
            weight = 1.0 / record['sigma']**2
            sum_weight, sum_value = grouped.get(key, (0.0, 0.0))
            grouped[key] = (sum_weight + weight,
                            sum_value + weight*record['value'])

        def loss(g):
            return sum(weight*(coefficient(u*g, n, ratio) - value/weight)**2
                       for (n,u,ratio),(weight,value) in grouped.items())

        result = minimize_scalar(loss, bounds=(0.7,1.4), method='bounded',
                                 options={'xatol': 1e-11})
        self.coupling = float(result.x)
        return self

    def predict(self,experiments):
        return predict_at(experiments,self.coupling)


@lru_cache(maxsize=None)
def nodes(order):
    x,w = roots_legendre(order)
    return (x+1)/2,w/2


def pair_fourier_coefficients(gamma, rho, terms):
    """Fourier coefficients of |1-rho*exp(i*theta)|**gamma.

    The hypergeometric expression sums the two binomial series analytically.
    Its last axis is the Fourier mode, including mode zero.
    """
    rho = np.asarray(rho)
    modes = np.arange(terms + 1)
    prefactor = np.empty(terms + 1)
    prefactor[0] = 1.0
    for k in range(1, terms + 1):
        prefactor[k] = prefactor[k-1] * (-gamma/2 + k-1) / k
    return (prefactor * rho[...,None]**modes *
            hyp2f1(modes-gamma/2, -gamma/2, modes+1,
                   rho[...,None]**2))


def three_ion_coefficient(gamma,ratio,order=32,terms=32):
    length = np.log(ratio)
    x,w = nodes(order)
    d1 = np.broadcast_to((length*x)[:,None],(order,order)).ravel()
    d2 = ((length-length*x)[:,None]*x[None,:]).ravel()
    weights = (length*(length-length*x)[:,None]*w[:,None]*w[None,:]).ravel()
    # With theta_1 fixed, the pair angles are alpha, beta, and alpha+beta.
    # Averaging their product keeps only equal Fourier modes in all three
    # factors. The former product of three angular averages discarded these
    # correlated modes.
    c12 = pair_fourier_coefficients(gamma, np.exp(-d1), terms)
    c23 = pair_fourier_coefficients(gamma, np.exp(-d2), terms)
    c13 = pair_fourier_coefficients(gamma, np.exp(-d1-d2), terms)
    angular = c12[:,0]*c23[:,0]*c13[:,0] + 2*np.sum(
        c12[:,1:]*c23[:,1:]*c13[:,1:], axis=1)
    span = length-d1-d2
    slope = 6-6*gamma
    weight = weights*angular*np.exp((4-3*gamma)*d1+(2-gamma)*d2)
    partition = np.dot(weight,span*exprel(slope*span))
    boundary = np.dot(weight,np.exp(slope*span))
    return -np.expm1(-2*length)*boundary/(6*partition)
