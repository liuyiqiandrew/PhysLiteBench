"""Independent finite oscillator-reservoir normal-mode calorimetry."""
from functools import lru_cache
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.linalg import eigh

TRUE_PARAMETER = 1.04


def experiment(temperature, damping=0., cutoff=1.):
    return dict(temperature=temperature, damping=damping, cutoff=cutoff)


def mode_capacity(omega, temperature):
    x = np.asarray(omega)/temperature
    z = np.exp(-x)
    return x*x*z/(-np.expm1(-x))**2


@lru_cache(None)
def bath_modes(frequency, damping, cutoff, order=20, extent=64):
    maximum = extent*max(cutoff, frequency, damping)
    edges = [0., .025]
    while edges[-1] < maximum:
        edges.append(min(2*edges[-1], maximum))
    nodes, weights = leggauss(order)
    omega = np.concatenate([(a+b)/2+(b-a)*nodes/2
                            for a,b in zip(edges[:-1],edges[1:])])
    weights = np.concatenate([(b-a)*weights/2
                              for a,b in zip(edges[:-1],edges[1:])])
    spectral = damping*omega*cutoff**2/(cutoff**2+omega**2)
    coupling = np.sqrt(2/np.pi*spectral*omega*weights)
    matrix = np.diag(np.r_[frequency**2+np.sum((coupling/omega)**2), omega**2])
    matrix[0,1:] = -coupling
    matrix[1:,0] = -coupling
    values = eigh(matrix, eigvals_only=True, check_finite=False)
    if values.min() <= 0:
        raise ValueError('Finite reservoir stiffness lost positivity')
    return np.sqrt(values), omega


def heat_capacity(e, frequency, order=20, extent=64):
    if e['damping'] == 0:
        return float(mode_capacity(frequency, e['temperature']))
    coupled, bare = bath_modes(frequency, e['damping'], e['cutoff'], order, extent)
    return float(np.sum(mode_capacity(coupled, e['temperature']))
                 -np.sum(mode_capacity(bare, e['temperature'])))


def predict(experiments, frequency=TRUE_PARAMETER):
    return np.array([heat_capacity(e, frequency) for e in experiments])


def calibration_inputs():
    return [experiment(T,0.,c) for _ in range(6)
            for T in [.16,.2,.25,.32,.42,.55,.72,.95] for c in [1.,2.,4.]]


def hidden_inputs():
    return {
        'temperature_sweep':[experiment(T,3.,1.) for T in [.18,.24,.32,.4]],
        'cutoff_sweep':[experiment(.32,2.5,c) for c in [1.,1.3,1.7,2.]],
        'coupling_sweep':[experiment(.27,g,1.25) for g in [1.5,2.,2.5,3.]],
        'uncoupled_anchors':[experiment(T,0.,c) for T,c in [(.12,1.),(.3,2.),(.9,4.)]]
    }
