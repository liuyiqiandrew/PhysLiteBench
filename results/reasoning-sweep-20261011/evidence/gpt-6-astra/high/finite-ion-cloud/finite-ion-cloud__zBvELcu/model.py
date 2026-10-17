"""Canonical finite-annulus pressure for one, two, or three counterions.

With L = log(R/a), phi = (1-exp(-2L)) * d(log Z)/dL / (2N).
Scale factors involving a cancel from this derivative and from phi.
"""

import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize_scalar
from scipy.special import exprel, hyp2f1, roots_legendre
from functools import lru_cache


@lru_cache(maxsize=4096)
def coefficient(gamma,count,ratio):
    if count == 3:
        return three_ion_coefficient(gamma,ratio)
    length = np.log(ratio)
    if count == 1:
        slope = 2-gamma
        return -np.expm1(-2*length)*np.exp(slope*length)/(2*length*exprel(slope*length))
    if count != 2:
        raise ValueError('ion_count must be 1, 2, or 3')
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
        """Fit the shared coupling by Gaussian maximum likelihood."""
        records = list(records)
        if not records:
            raise ValueError('At least one calibration record is required')
        experiments = [r['input'] for r in records]
        values = np.asarray([r['value'] for r in records], dtype=float)
        sigma = np.asarray([r['sigma'] for r in records], dtype=float)
        if not (np.isfinite(values).all() and np.isfinite(sigma).all()
                and (sigma > 0).all()):
            raise ValueError('Calibration values must be finite and sigma positive')

        def loss(coupling):
            residual = (predict_at(experiments, coupling)-values)/sigma
            return np.dot(residual, residual)

        result = minimize_scalar(loss, bounds=(0.7, 1.4), method='bounded',
                                 options={'xatol': 1e-11})
        if not result.success:
            raise RuntimeError('Coupling fit failed: ' + result.message)
        # A constrained optimum may lie exactly on either endpoint.
        self.coupling = float(min((0.7, result.x, 1.4), key=loss))
        return self

    def predict(self,experiments):
        if self.coupling is None:
            raise ValueError('Fit the model before predicting')
        return predict_at(experiments,self.coupling)


@lru_cache(maxsize=None)
def nodes(order):
    x,w = roots_legendre(order)
    return (x+1)/2,w/2


def _three_ion_angular(gamma, d1, d2, modes=48):
    """Joint angular average, retaining the closed triangle of pair angles.

    For f(t,theta) = |1-t*exp(i*theta)|**gamma, its Fourier coefficient is
    c_m(t) = (-p)_m/m! * t**m * 2F1(-p,m-p;m+1;t*t), p=gamma/2.
    Integrating both independent angles gives sum_m c_m(t12)c_m(t23)c_m(t13),
    with negative modes equal to positive ones. Multiplying only the three
    c_0 terms would incorrectly treat the pair angles as independent.
    """
    t = np.exp(-np.array([d1, d2, d1+d2]))
    p = gamma/2
    angular = np.prod(hyp2f1(-p, -p, 1, t*t), axis=0)
    prefactor = 1.0
    for m in range(1, modes+1):
        prefactor *= (m-1-p)/m
        if prefactor == 0:
            break  # At even integer gamma the Fourier series terminates.
        c = prefactor*t**m*hyp2f1(-p, m-p, m+1, t*t)
        angular += 2*np.prod(c, axis=0)
    return angular


def three_ion_coefficient(gamma,ratio,order=32):
    """Integrate ordered log radii, analytically integrating the inner one.

    The gaps d1,d2 occupy a triangle. For fixed gaps, the innermost log
    radius runs from zero to span=L-d1-d2. Its integral is span*exprel(s*span).
    Differentiating Z with respect to L replaces that integral by exp(s*span).
    The moving edge of the gap triangle contributes zero since span=0 there.
    """
    length = np.log(ratio)
    x,w = nodes(order)
    d1 = np.broadcast_to((length*x)[:,None],(order,order)).ravel()
    d2 = ((length-length*x)[:,None]*x[None,:]).ravel()
    weights = (length*(length-length*x)[:,None]*w[:,None]*w[None,:]).ravel()
    angular = _three_ion_angular(gamma, d1, d2)
    span = length-d1-d2
    slope = 6-6*gamma
    weight = weights*angular*np.exp((4-3*gamma)*d1+(2-gamma)*d2)
    partition = np.dot(weight,span*exprel(slope*span))
    boundary = np.dot(weight,np.exp(slope*span))
    return -np.expm1(-2*length)*boundary/(6*partition)
