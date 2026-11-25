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
        experiments = [record['input'] for record in records]
        observed = np.asarray([record['value'] for record in records], dtype=float)
        sigma = np.asarray([record['sigma'] for record in records], dtype=float)

        def objective(g):
            residual = (predict_at(experiments, g) - observed) / sigma
            return np.dot(residual, residual)

        result = minimize_scalar(objective, bounds=(0.7, 1.4), method='bounded',
                                 options={'xatol': 1e-10})
        if not result.success:
            raise RuntimeError('Coupling fit did not converge')
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
    # The three pair angles share only two independent ion angles.  The
    # constant Fourier mode of their product is a sum of products of equal
    # Fourier modes, rather than the product of the three angular means.
    modes = 256
    theta = 2*np.pi*np.arange(modes)/modes
    cosine = np.cos(theta)
    coefficients = []
    for gap in (d1, d2, d1+d2):
        q = np.exp(-gap)
        pair = np.maximum(1 - 2*q[:, None]*cosine + q[:, None]**2, 0)**(gamma/2)
        coefficients.append(np.fft.rfft(pair, axis=1).real / modes)
    angular = coefficients[0][:, 0]*coefficients[1][:, 0]*coefficients[2][:, 0]
    angular += 2*np.sum(coefficients[0][:, 1:-1]*coefficients[1][:, 1:-1]
                        *coefficients[2][:, 1:-1], axis=1)
    angular += coefficients[0][:, -1]*coefficients[1][:, -1]*coefficients[2][:, -1]
    span = length-d1-d2
    slope = 6-6*gamma
    weight = weights*angular*np.exp((4-3*gamma)*d1+(2-gamma)*d2)
    partition = np.dot(weight,span*exprel(slope*span))
    boundary = np.dot(weight,np.exp(slope*span))
    return -np.expm1(-2*length)*boundary/(6*partition)
