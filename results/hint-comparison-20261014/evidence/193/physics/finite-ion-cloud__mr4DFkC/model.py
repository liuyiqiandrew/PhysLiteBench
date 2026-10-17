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
                                  e['ion_count'],e['radius_ratio']) for e in experiments])


class Model:
    def __init__(self):
        self.coupling = None

    def fit(self,records):
        """Fit the common reference coupling by weighted least squares."""
        if not records:
            raise ValueError("at least one calibration record is required")

        experiments = [record['input'] for record in records]
        values = np.asarray([record['value'] for record in records], dtype=float)
        sigmas = np.asarray([record['sigma'] for record in records], dtype=float)
        if (not np.isfinite(values).all() or not np.isfinite(sigmas).all()
                or np.any(sigmas <= 0)):
            raise ValueError("calibration values must be finite and sigmas positive")

        def objective(g):
            residual = (predict_at(experiments, g) - values) / sigmas
            return float(np.dot(residual, residual))

        result = minimize_scalar(objective, bounds=(0.7, 1.4),
                                 method='bounded', options={'xatol': 1e-10})
        self.coupling = float(result.x)
        return self

    def predict(self,experiments):
        return predict_at(experiments,self.coupling)


@lru_cache(maxsize=None)
def nodes(order):
    x,w = roots_legendre(order)
    return (x+1)/2,w/2


def three_ion_coefficient(gamma,ratio,order=48,angular_modes=64):
    """Osmotic coefficient for three ions.

    The radial coordinates are ordered, with ``d1`` the lower radial gap
    and ``d2`` the upper one.  The three angular differences are not
    independent: if the first two are alpha and beta, the third is
    alpha+beta.  Their exact angular average is evaluated through Fourier
    coefficients, rather than multiplying three independent pair averages.
    """
    length = np.log(ratio)
    x,w = nodes(order)
    d1 = np.broadcast_to((length*x)[:,None],(order,order)).ravel()
    d2 = ((length-length*x)[:,None]*x[None,:]).ravel()
    weights = (length*(length-length*x)[:,None]*w[:,None]*w[None,:]).ravel()

    # If f_q(theta)=|1-q exp(i theta)|**gamma has Fourier coefficients
    # c_n(q), the constrained double angular integral is
    # sum_n c_n(q12)c_n(q23)c_n(q13).  For n >= 0,
    # c_n(q)=(-p)_n q^n/n! * 2F1(n-p,-p;n+1;q^2), p=gamma/2.
    p = gamma / 2.0
    gaps = (d2, d1, d1 + d2)
    q = [np.exp(-gap) for gap in gaps]
    n = np.arange(angular_modes + 1, dtype=float)[:, None]
    prefactor = poch(-p, n) / factorial(n)

    def fourier_coefficients(qi):
        return (prefactor * qi[None, :] ** n *
                hyp2f1(n - p, -p, n + 1, qi[None, :] ** 2))

    coefficients = [fourier_coefficients(qi) for qi in q]
    terms = coefficients[0] * coefficients[1] * coefficients[2]
    # n=0 occurs once; c_-n=c_n supplies the rest of the sum twice.
    angular = terms[0] + 2.0 * np.sum(terms[1:], axis=0)
    span = length-d1-d2
    slope = 6-6*gamma
    weight = weights*angular*np.exp((4-3*gamma)*d1+(2-gamma)*d2)
    partition = np.dot(weight,span*exprel(slope*span))
    boundary = np.dot(weight,np.exp(slope*span))
    return -np.expm1(-2*length)*boundary/(6*partition)
