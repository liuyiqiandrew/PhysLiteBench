from functools import lru_cache
import numpy as np
from scipy.optimize import minimize_scalar

MODES = np.arange(1, 7, dtype=float)


@lru_cache(None)
def quadrature():
    points, weights = np.polynomial.legendre.leggauss(64)
    x = (points+1)*np.pi/2
    basis = np.sqrt(2/np.pi)*np.sin(x[:, None]*MODES)
    return x, weights*np.pi/2, basis


@lru_cache(None)
def modal_temperatures(temperature, contrast, wavenumber, phase):
    x, weights, basis = quadrature()
    local = temperature*(1+contrast*np.cos(wavenumber*x+phase))
    return np.sum((weights*local)[:, None]*basis**2, axis=0)


def inputs(experiments):
    weights = np.array([e['weights'] for e in experiments], dtype=float)
    frequency = np.array([e['frequency'] for e in experiments], dtype=float)
    temperatures = np.array([modal_temperatures(e['temperature'],e['contrast'],e['wavenumber'],e['phase']) for e in experiments])
    return weights, frequency, temperatures


def spectrum(weights, frequency, temperatures, friction):
    response = weights/(MODES**2+1j*friction*frequency[:, None])
    return 2*friction*np.sum(temperatures*abs(response)**2, axis=1)


class Model:
    def __init__(self):
        self.friction = None

    def fit(self, records):
        data = inputs([r['input'] for r in records])
        observed = np.array([r['value'] for r in records])
        sigma = np.array([r['sigma'] for r in records])
        def objective(friction):
            residual = (spectrum(*data, friction)-observed)/sigma
            return float(residual@residual)
        self.friction = float(minimize_scalar(objective,bounds=(.6,1.8),method='bounded',options={'xatol':1e-12}).x)
        return self

    def predict(self, experiments):
        return spectrum(*inputs(experiments), self.friction)
