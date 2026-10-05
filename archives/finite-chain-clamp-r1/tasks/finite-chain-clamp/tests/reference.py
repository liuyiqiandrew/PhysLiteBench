"""Independent angular Boltzmann and projected-characteristic quadratures."""
from functools import lru_cache
import numpy as np
from numpy.polynomial.legendre import leggauss

TRUE_PARAMETER = 1.06


@lru_cache(None)
def fourier_nodes(panels, order):
    z, w = leggauss(order)
    left = np.arange(panels)*np.pi
    q = (left[:, None]+np.pi/2*(1+z)[None, :]).ravel()
    weights = np.tile(np.pi*w/2, panels)
    return q, weights


@lru_cache(maxsize=8192)
def holding_force(links, length, temperature, extension, panels=256, order=32,
                  clamp_stiffness=None):
    q, weights = fourier_nodes(panels, order)
    characteristic = np.sinc(q/np.pi)**links
    if clamp_stiffness is not None:
        characteristic *= np.exp(-temperature*q*q/(2*clamp_stiffness*length*length))
    phase = q*extension/length
    normalization = np.dot(weights, characteristic*np.cos(phase))
    derivative = np.dot(weights, q*characteristic*np.sin(phase))
    return float(temperature/length*derivative/normalization)


@lru_cache(maxsize=8192)
def mean_extension(links, length, temperature, force):
    u, weights = leggauss(96)
    weights = weights*np.exp(force*length*u/temperature)
    return float(links*length*np.dot(weights, u)/weights.sum())


def predict(experiments, length, panels=256, order=32):
    values = []
    for e in experiments:
        if e['mode'] == 'force':
            values.append(mean_extension(e['links'], length, e['temperature'], e['force']))
        else:
            values.append(holding_force(e['links'], length, e['temperature'], e['extension'], panels, order))
    return np.array(values)


def calibration_inputs():
    return [{'mode':'force', 'links':n, 'temperature':t, 'force':f}
            for n in [4,8,12] for t in [.6,1.,1.4] for f in [.3,1.,3.]]


def hidden_inputs():
    groups = {}
    for n in [4,6,8]:
        groups[f'chain_{n}'] = [{'mode':'clamp', 'links':n, 'temperature':t, 'extension':n*x}
                              for t in [.7,1.3] for x in [.18,.35,.52,.6]]
    groups['force_anchor'] = [{'mode':'force', 'links':n, 'temperature':t, 'force':f}
                              for n,t,f in [(5,.7,.5),(7,1.2,2.),(10,.9,.8),(11,1.3,2.5)]]
    return groups
