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
        if (not np.all(np.isfinite(values)) or not np.all(np.isfinite(sigmas))
                or np.any(sigmas <= 0)):
            raise ValueError('Calibration values must be finite and sigmas positive.')

        def objective(coupling):
            residual = (predict_at(experiments, coupling) - values) / sigmas
            return float(np.dot(residual, residual))

        result = minimize_scalar(objective, bounds=(0.7, 1.4), method='bounded',
                                 options={'xatol': 1e-12})
        candidates = (0.7, result.x, 1.4)
        self.coupling = min(candidates, key=objective)
        return self

    def predict(self,experiments):
        if self.coupling is None:
            raise ValueError('Call fit(records) before predict(experiments).')
        return predict_at(experiments,self.coupling)


@lru_cache(maxsize=None)
def nodes(order):
    x,w = roots_legendre(order)
    return (x+1)/2,w/2


def three_ion_angular_factor(gamma,d1,d2,order=1024,chunk_size=128):
    """Angular average of all three pair factors for ordered radii.

    For radial gaps d1 and d2, each pair contributes
    (1 - 2*rho*cos(theta) + rho**2)**(gamma/2), with rho=e**(-gap).
    The angular differences are coupled, so evaluate the joint average using
    the common Fourier modes of the three periodic factors.
    """
    d1 = np.asarray(d1, dtype=float)
    d2 = np.asarray(d2, dtype=float)
    gaps = (d1, d2, d1 + d2)
    base = 4.0 * np.sin(np.pi * np.arange(order) / order)**2
    result = np.empty(d1.size, dtype=float)

    for start in range(0, d1.size, chunk_size):
        stop = min(start + chunk_size, d1.size)
        spectra = []
        for gap in gaps:
            rho = np.exp(-gap[start:stop])[:, None]
            squared_distance = (1.0-rho)**2 + rho*base[None, :]
            values = squared_distance**(gamma/2.0)
            spectra.append(np.fft.fft(values, axis=1) / order)
        result[start:stop] = np.sum(spectra[0]*spectra[1]*spectra[2], axis=1).real
    return result


def three_ion_coefficient(gamma,ratio,order=48):
    length = np.log(ratio)
    x,w = nodes(order)
    d1 = np.broadcast_to((length*x)[:,None],(order,order)).ravel()
    d2 = ((length-length*x)[:,None]*x[None,:]).ravel()
    weights = (length*(length-length*x)[:,None]*w[:,None]*w[None,:]).ravel()
    angular = three_ion_angular_factor(gamma,d1,d2)
    span = length-d1-d2
    slope = 6-6*gamma
    weight = weights*angular*np.exp((4-3*gamma)*d1+(2-gamma)*d2)
    partition = np.dot(weight,span*exprel(slope*span))
    boundary = np.dot(weight,np.exp(slope*span))
    return -np.expm1(-2*length)*boundary/(6*partition)
