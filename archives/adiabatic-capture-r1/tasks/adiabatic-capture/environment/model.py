from functools import lru_cache
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.optimize import brentq


@lru_cache(None)
def quadrature(order):
    return leggauss(order)


def potential(q, delta):
    return q**4/4 - q*q/2 + delta*q


def phase_volume(energy, delta, order=96):
    roots = np.roots([.25, 0., -.5, delta, -energy])
    roots = np.sort(roots.real[abs(roots.imag) < 1e-8])
    if len(roots) < 2:
        return 0.
    z, w = quadrature(order)
    theta = np.pi*z/2
    total = 0.
    for left, right in zip(roots[::2], roots[1::2]):
        q = (right+left)/2 + (right-left)*np.sin(theta)/2
        jacobian = (right-left)*np.cos(theta)*np.pi/4
        momentum = np.sqrt(np.maximum(0., 2*(energy-potential(q, delta))))
        total += 2*np.sum(w*jacobian*momentum)
    return float(total)


def energy_for_volume(volume, delta):
    extrema = np.sort(np.roots([1., 0., -1., delta]).real)
    lower = float(min(potential(extrema[[0, 2]], delta))) + 1e-12
    upper = 1.
    while phase_volume(upper, delta) < volume:
        upper = 2*upper+1
    return float(brentq(lambda energy: phase_volume(energy, delta)-volume,
                        lower, upper, xtol=2e-12))


@lru_cache(maxsize=16384)
def mean_energy(action_scale, delta, s_final, order=32):
    z, w = quadrature(order)
    action = action_scale*(1.1+.1*z)
    energy = [s_final*s_final*energy_for_volume(2*np.pi*j/s_final**1.5, delta)
              for j in action]
    return float(w@energy/2)


def predict_at(experiments, action_scale):
    return np.asarray([mean_energy(float(action_scale), float(e['delta']), float(e['s_final']))
                       for e in experiments])


class Model:
    def __init__(self):
        self.action_scale = None

    def fit(self, records):
        raise NotImplementedError('Fit the shared preparation scale from the calibration records.')

    def predict(self, experiments):
        return predict_at(experiments, self.action_scale)
