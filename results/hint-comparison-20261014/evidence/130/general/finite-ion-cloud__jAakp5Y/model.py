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
        inputs = [record['input'] for record in records]
        values = np.asarray([record['value'] for record in records], dtype=float)
        sigmas = np.asarray([record['sigma'] for record in records], dtype=float)

        def objective(g):
            residual = (predict_at(inputs, g)-values)/sigmas
            return np.dot(residual, residual)

        result = minimize_scalar(objective, bounds=(0.7, 1.4), method='bounded',
                                 options={'xatol': 1e-11})
        self.coupling = float(np.clip(result.x, 0.7, 1.4))
        return self

    def predict(self,experiments):
        return predict_at(experiments,self.coupling)


@lru_cache(maxsize=None)
def nodes(order):
    x,w = roots_legendre(order)
    return (x+1)/2,w/2


def three_ion_coefficient(gamma,ratio,order=32):
    """Finite-cell coefficient for three ions.

    After ordering the logarithmic radii, the remaining angular integral is
    not a product of three pair angular integrals: the three relative angles
    have one constraint.  If

        f_q(theta) = |1-q exp(i theta)|**gamma

    and c_n(q) are its Fourier coefficients, the normalized two-angle
    integral is sum_n c_n(q1)c_n(q2)c_n(q3).  This evaluates that angular
    integral before doing the two-dimensional radial quadrature.
    """
    length = np.log(ratio)
    x,w = nodes(order)
    d1 = np.broadcast_to((length*x)[:,None],(order,order)).ravel()
    d2 = ((length-length*x)[:,None]*x[None,:]).ravel()
    weights = (length*(length-length*x)[:,None]*w[:,None]*w[None,:]).ravel()
    angular = three_ion_angular(gamma, np.exp(-d1), np.exp(-d2),
                                np.exp(-d1-d2))
    span = length-d1-d2
    slope = 6-6*gamma
    weight = weights*angular*np.exp((4-3*gamma)*d1+(2-gamma)*d2)
    partition = np.dot(weight,span*exprel(slope*span))
    boundary = np.dot(weight,np.exp(slope*span))
    return -np.expm1(-2*length)*boundary/(6*partition)


def _fourier_coefficients(gamma, q, order=128):
    """Return c_n(q) for n=0,...,order-1, for each q in *q*.

    The binomial series gives, for n >= 0,

      c_n(q) = q**n (-gamma/2)_n/n!
               * 2F1(n-gamma/2,-gamma/2;n+1;q**2).

    The recurrence for the Pochhammer prefactor avoids factorial overflow.
    """
    q = np.asarray(q, dtype=float).ravel()
    n = np.arange(order)[:, None]
    a = -gamma/2

    prefactor = np.ones(order, dtype=float)
    if order > 1:
        prefactor[1:] = np.cumprod((a+np.arange(order-1))/np.arange(1, order))

    with np.errstate(over='ignore', invalid='ignore'):
        coefficients = (q[None, :]**n * prefactor[:, None] *
                        hyp2f1(a+n, a, n+1, q[None, :]**2))

    # For nonnegative even integer gamma, the binomial series terminates.
    # Explicitly clear the zero prefactors so that a harmless 0*NaN from a
    # high-order special-function evaluation cannot enter the quadrature.
    coefficients[prefactor == 0] = 0
    return coefficients


def three_ion_angular(gamma, q1, q2, q3, order=128):
    """Normalized angular integral for the three-ion ordered configuration."""
    q1, q2, q3 = (np.asarray(q, dtype=float) for q in (q1, q2, q3))
    shape = q1.shape
    coefficients = _fourier_coefficients(
        gamma, np.concatenate((q1.ravel(), q2.ravel(), q3.ravel())), order)
    c1, c2, c3 = np.split(coefficients, 3, axis=1)
    result = c1[0]*c2[0]*c3[0]
    result += 2*np.sum(c1[1:]*c2[1:]*c3[1:], axis=0)
    return result.reshape(shape)
