"""Tracer loss and axial transport in a circular tube."""
import numpy as np
from scipy.optimize import brentq
from scipy.special import j0, j1, jn_zeros, roots_legendre

EDGE = float(jn_zeros(0, 1)[0])
X, W = roots_legendre(96)
R = (X + 1)/2
WEIGHT = W*R/2


def radial_mode(diffusivity, radius, capture):
    if capture == 0:
        return 0.0
    beta = capture*radius/diffusivity
    return brentq(lambda q: q*j1(q)-beta*j0(q), 0, EDGE, xtol=1e-14)


def reading(experiment, diffusivity):
    radius = experiment['radius']
    q = radial_mode(diffusivity, radius, experiment['capture'])
    if experiment['observable'] == 'loss_rate':
        return diffusivity*q*q/radius**2
    plug, peak = experiment['plug'], experiment['peak']
    if peak == 0:
        return float(plug)
    density = j0(q*R)
    weight = WEIGHT*density
    velocity = plug+peak*(1-R*R)
    return float(np.dot(weight, velocity)/weight.sum())


def predict_at(experiments, diffusivity):
    cache = {}
    out = []
    for experiment in experiments:
        key = tuple(sorted(experiment.items()))
        if key not in cache:
            cache[key] = reading(experiment, diffusivity)
        out.append(cache[key])
    return np.array(out,dtype=float)


class Model:
    def __init__(self):
        self.diffusivity = None

    def fit(self, records):
        raise NotImplementedError('Fit diffusivity from the calibration records.')

    def predict(self, experiments):
        if self.diffusivity is None:
            raise RuntimeError('Call fit before predict.')
        return predict_at(experiments,self.diffusivity)
