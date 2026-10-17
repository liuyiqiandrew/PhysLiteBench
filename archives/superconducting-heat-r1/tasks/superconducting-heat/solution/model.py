import numpy as np
from functools import lru_cache
from numpy.polynomial.legendre import leggauss
from scipy.special import expit

KB = 1.380649e-23
ELECTRON_CHARGE = 1.602176634e-19
NODES, WEIGHTS = leggauss(160)


def gap(temperature, critical):
    return 1.764*critical*np.tanh(1.74*np.sqrt(critical/temperature-1))


@lru_cache(maxsize=512)
def power_per_conductance(left, right, phase):
    dl, dr = gap(left, 1.2), gap(right, 1.9)
    high, low = max(dl, dr), min(dl, dr)
    upper = np.sqrt((high+40*max(left, right))**2-high**2)
    s = (NODES+1)*upper/2
    energy = np.sqrt(high**2+s**2)
    occupation = expit(-energy/left)-expit(-energy/right)
    integrand = (energy**2-dl*dr*np.cos(phase))*occupation/np.sqrt(energy**2-low**2)
    integral = upper/2*(WEIGHTS@integrand)
    return 2e6*(KB/ELECTRON_CHARGE)**2*integral


def predict_at(experiments, conductance):
    return conductance*np.array([power_per_conductance(e['left_temperature'], e['right_temperature'], e['phase']) for e in experiments])


class Model:
    def __init__(self):
        self.conductance = None

    def fit(self, records):
        x = predict_at([r['input'] for r in records], 1.)
        y = np.array([r['value'] for r in records])
        sigma = np.array([r['sigma'] for r in records])
        self.conductance = float(np.clip(np.sum(x*y/sigma**2)/np.sum((x/sigma)**2), 20., 80.))
        return self

    def predict(self, experiments):
        return predict_at(experiments, self.conductance)
