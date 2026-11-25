"""Finite-cell osmotic coefficient for the two-dimensional ionic cell.

The radial coordinates are represented by ``s = log(r/a)``.  This removes
the (irrelevant) absolute disk radius and turns the derivative with respect
to cell area into an outer-boundary term in each configurational integral.
"""

from functools import lru_cache

import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize_scalar
from scipy.special import exprel, hyp2f1, roots_legendre


@lru_cache(maxsize=None)
def coefficient(gamma,count,ratio):
    """Return P/(kT N/A) at coupling ``gamma``.

    After angular integrations, configurations are ordered by radius.  The
    derivative of the radial partition function is its upper-boundary term.
    In particular, if ``Z(L)`` is the remaining dimensionless partition
    integral and ``L=log(R/a)``, then

        phi = (1 - exp(-2 L)) * d(log Z)/dL / (2 N).

    Factors independent of L, including the disk radius and permutations of
    the ordered particles, cancel from this expression.
    """
    gamma = float(gamma)
    count = int(count)
    ratio = float(ratio)
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
        """Fit the shared reference coupling by Gaussian weighted least squares."""
        records = list(records)
        if not records:
            raise ValueError('at least one calibration record is required')

        experiments = [record['input'] for record in records]
        observed = np.asarray([record['value'] for record in records], dtype=float)
        sigma = np.asarray([record['sigma'] for record in records], dtype=float)
        if (not np.isfinite(observed).all() or not np.isfinite(sigma).all()
                or np.any(sigma <= 0)):
            raise ValueError('calibration values must be finite and sigma must be positive')

        def loss(g):
            residual = (predict_at(experiments, g) - observed) / sigma
            return float(np.dot(residual, residual))

        # The bounds are part of the physical specification, rather than a
        # statistical prior, so retain a constrained one-parameter fit.
        result = minimize_scalar(loss, bounds=(0.7, 1.4), method='bounded',
                                 options={'xatol': 1e-11})
        # ``bounded`` searches the open interval; explicitly include the
        # physically allowed endpoints in case noisy data favour one.
        candidates = (0.7, float(result.x), 1.4)
        self.coupling = min(candidates, key=loss)
        return self

    def predict(self,experiments):
        if self.coupling is None:
            raise RuntimeError('fit must be called before predict')
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
