import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize_scalar
from scipy.special import exprel, hyp2f1, roots_legendre
from functools import lru_cache


@lru_cache(maxsize=None)
def coefficient(gamma,count,ratio):
    """Finite-cell osmotic coefficient at a fixed reduced coupling.

    The radial coordinates used below are ``t=log(r/a)``.  Differentiating
    the canonical partition function with respect to the outer boundary
    gives ``phi=(1-exp(-2L)) d(log Z)/dL/(2 N)``, with ``L=log(R/a)``.
    Constants in Z (including the disk radius) consequently cancel.
    """
    if count not in (1, 2, 3):
        raise ValueError("ion_count must be 1, 2, or 3")
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
        """Fit the shared reference coupling by Gaussian maximum likelihood.

        Each calibration uncertainty is an independent additive Gaussian
        standard deviation, so the negative log likelihood (up to a
        constant) is the weighted squared residual used here.
        """
        records = list(records)
        if not records:
            raise ValueError("at least one calibration record is required")

        experiments = [record['input'] for record in records]
        observed = np.asarray([record['value'] for record in records], dtype=float)
        sigma = np.asarray([record['sigma'] for record in records], dtype=float)
        if (not np.all(np.isfinite(observed)) or
                not np.all(np.isfinite(sigma)) or np.any(sigma <= 0)):
            raise ValueError("calibration values must be finite and sigmas positive")

        def loss(coupling):
            residual = (predict_at(experiments, coupling) - observed) / sigma
            return np.dot(residual, residual)

        result = minimize_scalar(
            loss, bounds=(0.7, 1.4), method='bounded',
            options={'xatol': 1e-11},
        )
        if not result.success or not np.isfinite(result.fun):
            raise RuntimeError("could not fit the electrostatic coupling")
        # ``bounded`` searches the open interval; explicitly include the
        # physically allowed endpoints in case a noisy calibration set puts
        # its maximum-likelihood estimate on a boundary.
        candidates = ((float(result.x), float(result.fun)),
                      (0.7, float(loss(0.7))),
                      (1.4, float(loss(1.4))))
        self.coupling = min(candidates, key=lambda candidate: candidate[1])[0]
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
    weight = weights*angular*np.exp((4-3*gamma)*d1+(2-gamma)*d2)
    partition = np.dot(weight,span*exprel(slope*span))
    boundary = np.dot(weight,np.exp(slope*span))
    return -np.expm1(-2*length)*boundary/(6*partition)
