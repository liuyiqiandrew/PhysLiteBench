from functools import lru_cache
import numpy as np
from numpy.polynomial.legendre import leggauss


@lru_cache(maxsize=1)
def quadrature():
    radial, radial_weight = leggauss(80)
    cosine, cosine_weight = leggauss(32)
    return radial, radial_weight, cosine, cosine_weight


@lru_cache(maxsize=1024)
def mean_energy(mass, temperature, gas_speed, analysis_speed):
    nodes, weights, cosine, cosine_weight = quadrature()
    maximum = np.arccosh(1 + 60*temperature/mass)
    rapidity = (nodes + 1)*maximum/2
    energy = mass*np.cosh(rapidity)
    momentum = mass*np.sinh(rapidity)
    radial_weight = weights*maximum/2*momentum**2*energy*np.exp(-(energy-mass)/temperature)
    rest_px = momentum[:, None]*cosine[None, :]
    gamma_gas = 1/np.sqrt(1-gas_speed**2)
    lab_energy = gamma_gas*(energy[:, None]+gas_speed*rest_px)
    lab_px = gamma_gas*(rest_px+gas_speed*energy[:, None])
    observed = (lab_energy-analysis_speed*lab_px)/np.sqrt(1-analysis_speed**2)
    measure = radial_weight[:, None]*cosine_weight[None, :]
    return float(np.sum(measure*observed)/np.sum(measure))


def predict_at(experiments, mass):
    return np.asarray([mean_energy(float(mass),float(e['temperature']),
                                  float(e['gas_speed']),float(e['analysis_speed']))
                       for e in experiments],dtype=float)


class Model:
    def __init__(self):
        self.mass = None

    def fit(self, records):
        from scipy.optimize import minimize_scalar
        experiments = [r['input'] for r in records]
        values = np.asarray([r['value'] for r in records])
        sigma = np.asarray([r['sigma'] for r in records])
        result = minimize_scalar(lambda m: np.sum(((predict_at(experiments,m)-values)/sigma)**2),
                                 bounds=(.8,1.2),method='bounded',options={'xatol':1e-12})
        self.mass = float(result.x)
        return self

    def predict(self, experiments):
        return predict_at(experiments, self.mass)
