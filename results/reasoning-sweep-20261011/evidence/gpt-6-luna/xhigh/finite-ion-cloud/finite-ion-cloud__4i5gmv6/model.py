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
        records = list(records)
        if not records:
            raise ValueError('At least one calibration record is required.')

        experiments = [record['input'] for record in records]
        values = np.asarray([record['value'] for record in records], dtype=float)
        sigmas = np.asarray([record['sigma'] for record in records], dtype=float)
        if np.any(~np.isfinite(values)) or np.any(~np.isfinite(sigmas)) or np.any(sigmas <= 0):
            raise ValueError('Calibration values must be finite and sigmas must be positive.')

        def weighted_residual_sum(coupling):
            residual = (predict_at(experiments, coupling) - values) / sigmas
            return float(np.dot(residual, residual))

        result = minimize_scalar(
            weighted_residual_sum,
            bounds=(0.7, 1.4),
            method='bounded',
            options={'xatol': 1e-10},
        )
        if not result.success or not np.isfinite(result.fun):
            raise RuntimeError('Could not fit the reference electrostatic coupling.')
        candidates = [(result.fun, result.x)]
        candidates.extend((weighted_residual_sum(bound), bound) for bound in (0.7, 1.4))
        loss, coupling = min(candidates)
        if not np.isfinite(loss):
            raise RuntimeError('Could not fit the reference electrostatic coupling.')
        self.coupling = float(coupling)
        return self

    def predict(self,experiments):
        return predict_at(experiments,self.coupling)


@lru_cache(maxsize=None)
def nodes(order):
    x,w = roots_legendre(order)
    return (x+1)/2,w/2


def three_ion_angular_average(gamma,gap1,gap2,terms=32):
    """Angular average of all three pair factors for ordered ion radii.

    If the successive log-radius gaps are gap1 and gap2, the three radial
    ratios are exp(-gap1), exp(-gap2), and exp(-(gap1+gap2)). The angle
    differences around the triangle are coupled, so their averages cannot
    be multiplied independently. Fourier expansion leaves a single mode
    sum because the three angular integrations enforce matching modes.
    """
    p = gamma/2
    x1, x2 = np.exp(-gap1), np.exp(-gap2)
    x3 = np.exp(-(gap1+gap2))
    x1sq, x2sq, x3sq = x1*x1, x2*x2, x3*x3

    c01 = hyp2f1(-p,-p,1,x1sq)
    c02 = hyp2f1(-p,-p,1,x2sq)
    c03 = hyp2f1(-p,-p,1,x3sq)
    average = c01*c02*c03

    # Fourier coefficient k of |1-x*exp(i*theta)|**gamma is
    # x**k*(-p)_k/k! * 2F1(k-p,-p;k+1;x**2), for p=gamma/2.
    scale = 1.0
    for k in range(1,terms+1):
        scale *= (k-1-p)/k
        if scale == 0:
            break
        c1 = scale*x1**k*hyp2f1(k-p,-p,k+1,x1sq)
        c2 = scale*x2**k*hyp2f1(k-p,-p,k+1,x2sq)
        c3 = scale*x3**k*hyp2f1(k-p,-p,k+1,x3sq)
        average += 2*c1*c2*c3
    return average


def three_ion_coefficient(gamma,ratio,order=32):
    length = np.log(ratio)
    x,w = nodes(order)
    d1 = np.broadcast_to((length*x)[:,None],(order,order)).ravel()
    d2 = ((length-length*x)[:,None]*x[None,:]).ravel()
    weights = (length*(length-length*x)[:,None]*w[:,None]*w[None,:]).ravel()
    angular = three_ion_angular_average(gamma,d1,d2)
    span = length-d1-d2
    slope = 6-6*gamma
    weight = weights*angular*np.exp((4-3*gamma)*d1+(2-gamma)*d2)
    partition = np.dot(weight,span*exprel(slope*span))
    boundary = np.dot(weight,np.exp(slope*span))
    return -np.expm1(-2*length)*boundary/(6*partition)
