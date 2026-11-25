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
        """Estimate the common reference coupling by Gaussian likelihood.

        The radius of the disk is deliberately absent from ``predict_at``:
        after scaling all positions by ``a``, it contributes only a factor to
        the configurational integral which is independent of the outer wall.
        It therefore cannot affect the pressure coefficient.
        """
        records = list(records)
        if not records:
            raise ValueError("at least one calibration record is required")

        experiments = [record["input"] for record in records]
        values = np.asarray([record["value"] for record in records], dtype=float)
        sigmas = np.asarray([record["sigma"] for record in records], dtype=float)
        if (not np.isfinite(values).all() or not np.isfinite(sigmas).all()
                or np.any(sigmas <= 0)):
            raise ValueError("calibration values must be finite and sigmas positive")

        def chi_square(g):
            residual = (predict_at(experiments, g) - values) / sigmas
            return float(np.dot(residual, residual))

        result = minimize_scalar(
            chi_square, bounds=(0.7, 1.4), method="bounded",
            options={"xatol": 1e-10},
        )
        if not result.success or not np.isfinite(result.fun):
            raise RuntimeError("could not fit electrostatic coupling")
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
    angular = np.ones(len(d1))
    for gap in [d1,d2,d1+d2]:
        angular *= hyp2f1(-gamma/2,-gamma/2,1,np.exp(-2*gap))
    span = length-d1-d2
    slope = 6-6*gamma
    # d1 is the gap from the outer to the middle ion, and d2 the gap
    # from the middle to the inner ion.  On ordering the three radii, the
    # radial Jacobian and electrostatic energy give their respective
    # coefficients 2-gamma and 4-3*gamma.  (They are not interchangeable.)
    weight = weights*angular*np.exp((2-gamma)*d1+(4-3*gamma)*d2)
    partition = np.dot(weight,span*exprel(slope*span))
    boundary = np.dot(weight,np.exp(slope*span))
    return -np.expm1(-2*length)*boundary/(6*partition)
