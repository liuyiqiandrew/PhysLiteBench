from functools import lru_cache
import numpy as np
from scipy.optimize import brentq
from scipy.special import expit


@lru_cache(1)
def _grid():
    nodes, weights = np.polynomial.legendre.leggauss(192)
    angles = np.pi*(nodes+1)/2
    return angles, weights*np.pi/2


@lru_cache(512)
def _occupation(energy, temperature, contact):
    angles, weights = _grid()
    band_energy = -2*np.cos(angles)
    sine = np.sin(angles)
    filling = expit(-band_energy/temperature)
    surface = (band_energy-2j*sine)/2
    spectral = -np.imag(1/(band_energy-energy-contact**2*surface))/np.pi
    result = float(weights@(2*sine*spectral*filling))
    def surface_outside(value):
        return (value-np.sign(value)*np.sqrt(value**2-4))/2
    def denominator(value):
        return value-energy-contact**2*surface_outside(value)
    poles = []
    if contact**2 > 2-energy:
        poles.append(brentq(denominator, 2+1e-12, 2+abs(energy)+2*contact))
    if contact**2 > 2+energy:
        poles.append(brentq(denominator, -2-abs(energy)-2*contact, -2-1e-12))
    for pole in poles:
        derivative = (1-abs(pole)/np.sqrt(pole**2-4))/2
        weight = 1/(1-contact**2*derivative)
        projection = float(weights@(2*sine**2/np.pi*filling/(pole-band_energy)**2))
        population = weight*(1+contact**2*projection)
        result += weight*population
    return float(result)


class Model:
    def __init__(self):
        self.coupling_scale = .8

    def fit(self, records):
        from scipy.optimize import minimize_scalar
        inputs = [r['input'] for r in records]
        values = np.asarray([r['value'] for r in records])
        sigmas = np.asarray([r['sigma'] for r in records])
        def loss(scale):
            predicted = np.asarray([_occupation(e['orbital_energy'], e['temperature'],
                                    scale*e['contact_multiplier']) for e in inputs])
            return float(np.sum(((predicted-values)/sigmas)**2))
        self.coupling_scale = float(minimize_scalar(loss, bounds=(.65,.95),
                                                   method='bounded', options={'xatol':1e-12}).x)
        return self

    def predict(self, experiments):
        return np.asarray([_occupation(e['orbital_energy'], e['temperature'],
                           self.coupling_scale*e['contact_multiplier']) for e in experiments])
