from functools import lru_cache
import numpy as np
from numpy.polynomial.legendre import leggauss


@lru_cache(maxsize=128)
def pulse_grid(center, width):
    u, weights = leggauss(96)
    weights = weights*(1-u*u)**8
    return center+width*u, weights/weights.sum()


def indices(omega, strength):
    denominator = 1-omega*omega
    phase = 1+strength/denominator
    group = 1+strength*(1+omega*omega)/denominator**2
    return phase, group


def predict_at(experiments, strength):
    values = []
    for e in experiments:
        omega, weights = pulse_grid(e['center'], e['width'])
        phase, group = indices(omega, strength)
        if e['mode'] == 'transit':
            value = e['length']*np.dot(weights, group)
        else:
            value = e['length']*np.dot(weights, (1-phase)*group)
        values.append(value)
    return np.asarray(values, dtype=float)


class Model:
    def __init__(self):
        self.oscillator_strength = None

    def fit(self, records):
        experiments = [r['input'] for r in records]
        values = np.asarray([r['value'] for r in records])
        sigma = np.asarray([r['sigma'] for r in records])
        lower = predict_at(experiments, .2)
        slope = (predict_at(experiments, .4)-lower)/.2
        delta = np.dot(slope/sigma, (values-lower)/sigma)/np.dot(slope/sigma, slope/sigma)
        self.oscillator_strength = float(np.clip(.2+delta, .2, .4))
        return self

    def predict(self, experiments):
        return predict_at(experiments, self.oscillator_strength)
