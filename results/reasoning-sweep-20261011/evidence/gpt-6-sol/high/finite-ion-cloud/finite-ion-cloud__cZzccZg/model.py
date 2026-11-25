import numpy as np
from scipy.fft import rfft
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
        # The radius a sets the length scale and cancels from phi. Combine
        # repeated preparations before evaluating the likelihood.
        grouped = {}
        for record in records:
            experiment = record['input']
            key = (experiment['ion_count'],
                   experiment['temperature_multiplier'],
                   experiment['radius_ratio'])
            weight = 1.0 / record['sigma']**2
            sum_weight, sum_value = grouped.get(key, (0.0, 0.0))
            grouped[key] = (sum_weight + weight,
                            sum_value + weight * record['value'])

        def loss(coupling):
            return sum(weight * (coefficient(u * coupling, count, ratio)
                                 - weighted_value / weight)**2
                       for (count, u, ratio), (weight, weighted_value)
                       in grouped.items())

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


@lru_cache(maxsize=None)
def angular_grid(order):
    angle = np.arange(order) * (np.pi / order)
    return 4 * np.sin(angle)**2


def pair_fourier(gamma, gaps, order):
    """Fourier coefficients of |1-exp(-gap+i*theta)|**gamma."""
    t = np.exp(-np.asarray(gaps))[:, None]
    kernel = ((1-t)**2 + t*angular_grid(order)[None, :])**(gamma/2)
    return rfft(kernel, axis=1).real / order


def three_ion_coefficient(gamma,ratio,order=64,angular_order=512):
    """Finite-cell contact density with all three angular pairs coupled."""
    length = np.log(ratio)
    x,w = nodes(order)
    first = length*x
    d1 = np.broadcast_to(first[:,None],(order,order)).ravel()
    d2 = ((length-first)[:,None]*x[None,:]).ravel()
    weights = (length*(length-first)[:,None]*w[:,None]*w[None,:]).ravel()

    # The same two relative angles occur in all three pair distances. In
    # their Fourier expansions, angular integration keeps modes with the
    # same index in each of the three factors.
    c12 = np.repeat(pair_fourier(gamma, first, angular_order), order, axis=0)
    c23 = pair_fourier(gamma, d2, angular_order)
    c13 = pair_fourier(gamma, d1+d2, angular_order)
    product = c12*c23*c13
    angular = product[:,0] + 2*np.sum(product[:,1:-1],axis=1) + product[:,-1]
    span = length-d1-d2
    slope = 6-6*gamma
    weight = weights*angular*np.exp((4-3*gamma)*d1+(2-gamma)*d2)
    partition = np.dot(weight,span*exprel(slope*span))
    boundary = np.dot(weight,np.exp(slope*span))
    return -np.expm1(-2*length)*boundary/(6*partition)
