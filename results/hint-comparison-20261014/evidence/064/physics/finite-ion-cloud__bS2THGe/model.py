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
        values = np.asarray([record['value'] for record in records], dtype=float)
        sigmas = np.asarray([record['sigma'] for record in records], dtype=float)
        experiments = [record['input'] for record in records]

        def objective(g):
            residual = (predict_at(experiments, g) - values) / sigmas
            return np.mean(residual*residual)

        result = minimize_scalar(objective, bounds=(0.7, 1.4), method='bounded',
                                 options={'xatol': 1e-12})
        self.coupling = float(np.clip(result.x, 0.7, 1.4))
        return self

    def predict(self,experiments):
        return predict_at(experiments,self.coupling)


@lru_cache(maxsize=None)
def nodes(order):
    x,w = roots_legendre(order)
    return (x+1)/2,w/2


def three_ion_coefficient(gamma,ratio,order=32):
    length = np.log(ratio)
    # The angular factors for the three pairs are not independent.  With
    # sorted log-radii, d1 and d2 are the adjacent gaps and the two
    # independent angle differences are alpha and beta.  Thus the angular
    # average is
    #
    # < f(d1,alpha) f(d2,beta) f(d1+d2,alpha+beta) >_{alpha,beta},
    #
    # rather than the product of three one-pair averages.
    #
    # A power map puts quadrature points near d1=0 and d2=0, where the
    # pair-distance factors have their weakest smoothness.  The third power
    # is inexpensive and makes the radial result effectively converged over
    # the requested domain.
    x,w = nodes(order)
    radial_power = 3
    mapped = x**radial_power
    mapped_weights = w*radial_power*x**(radial_power-1)
    d1 = np.broadcast_to((length*mapped)[:,None],(order,order)).ravel()
    d2 = ((length-length*mapped)[:,None]*mapped[None,:]).ravel()
    weights = (length*(length-length*mapped)[:,None]
               *mapped_weights[:,None]*mapped_weights[None,:]).ravel()

    # For f(d,theta)=(1-2 exp(-d) cos(theta)+exp(-2d))**(gamma/2),
    # multiplication of Fourier coefficients gives the shared two-angle
    # average.  A periodic trapezoidal grid is particularly effective here;
    # 1024 points also resolves the weak-coupling endpoint of the domain.
    angular_order = 1024
    theta = 2*np.pi*np.arange(angular_order)/angular_order
    cosine = np.cos(theta)
    power = gamma/2

    def fourier_coefficients(gaps):
        z = np.exp(-gaps)
        base = (1 - 2*z[:,None]*cosine + z[:,None]**2)
        # Roundoff can make the exactly nonnegative expression very slightly
        # negative at theta=0 for the smallest gaps.
        return np.fft.fft(np.maximum(base, 0)**power, axis=1)/angular_order

    c1 = fourier_coefficients(d1)
    c2 = fourier_coefficients(d2)
    c3 = fourier_coefficients(d1+d2)
    angular = np.sum(c1*c2*c3, axis=1).real

    span = length-d1-d2
    slope = 6-6*gamma
    weight = weights*angular*np.exp((4-3*gamma)*d1+(2-gamma)*d2)
    partition = np.dot(weight,span*exprel(slope*span))
    boundary = np.dot(weight,np.exp(slope*span))
    return -np.expm1(-2*length)*boundary/(6*partition)
