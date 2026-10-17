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
        # The disk radius drops out after scaling every position by a.  Group
        # repeated preparations so the likelihood needs one prediction each.
        preparations = list(dict.fromkeys(
            (r['input']['temperature_multiplier'],
             r['input']['ion_count'], r['input']['radius_ratio'])
            for r in records))
        index = {p: i for i, p in enumerate(preparations)}
        record_index = np.array([
            index[(r['input']['temperature_multiplier'],
                   r['input']['ion_count'], r['input']['radius_ratio'])]
            for r in records], dtype=int)
        values = np.array([r['value'] for r in records])
        sigmas = np.array([r['sigma'] for r in records])

        def negative_log_likelihood(g):
            predictions = np.array([
                coefficient(g*u, count, ratio)
                for u, count, ratio in preparations])
            return np.sum(((predictions[record_index]-values)/sigmas)**2)

        result = minimize_scalar(negative_log_likelihood,
                                 bounds=(0.7, 1.4), method='bounded',
                                 options={'xatol': 1e-11})
        self.coupling = float(result.x)
        return self

    def predict(self,experiments):
        return predict_at(experiments,self.coupling)


@lru_cache(maxsize=None)
def nodes(order):
    x,w = roots_legendre(order)
    return (x+1)/2,w/2


def three_ion_coefficient(gamma,ratio,order=32):
    # Order the logarithmic radii as s, s+d1, s+d1+d2.  The integral over
    # s is analytic; differentiating its upper limit gives the wall pressure.
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


def three_ion_angular_average(gamma,d1,d2):
    """Average all three pair distances over the *same* three ion angles.

    The Fourier coefficient of |1-z exp(i theta)|**gamma at mode m is
    (-gamma/2)_m z**m / m! times a Gauss hypergeometric function.  Angle
    integration selects the same Fourier mode for all three pairs.
    """
    gaps = (d1,d2,d1+d2)
    z2 = [np.exp(-2*gap) for gap in gaps]
    z = [np.exp(-gap) for gap in gaps]
    power = -gamma/2
    angular = np.prod([hyp2f1(power,power,1,q) for q in z2],axis=0)
    factor = 1.0
    for mode in range(1,49):
        factor *= (power+mode-1)/mode
        terms = [factor * zz**mode *
                 hyp2f1(mode+power,power,mode+1,q)
                 for zz,q in zip(z,z2)]
        angular += 2*terms[0]*terms[1]*terms[2]
    return angular
