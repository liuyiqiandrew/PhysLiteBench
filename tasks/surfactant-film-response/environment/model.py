import numpy as np
from scipy.optimize import minimize_scalar


def fractions(e, rate):
    """Single-molecule fractions across the camera's speed range."""
    low = (1 + e['lo'] / rate) / 2
    high = (1 - e['hi'] / rate) / 2
    return {'L': low, 'M': 1-low-high, 'H': high, '*': 1.0}


def response(e, rate):
    if e['mode'] == 'bands':
        p = fractions(e, rate)
        return np.prod([p[c] for c in e['word']])
    # Integrate over speeds selected by the reference molecule's window.
    x, weight = np.polynomial.legendre.leggauss(8)
    velocity = (e['lo'] + e['hi'] + x*(e['hi']-e['lo'])) / 2
    low = (1 + velocity/rate) / 2
    probability = np.ones_like(x)
    for c in e['word'][:-1]:
        probability *= {'L': low, 'H': 1-low, 'M': 0., '*': 1.}[c]
    return weight @ probability / 2


class Model:
    def fit(self, records):
        def cost(rate):
            return sum((fractions(row['input'], rate)[row['input']['word']] - row['value'])**2
                       for row in records)
        self.rate = float(minimize_scalar(cost, bounds=(.85, 1.15), method='bounded',
                                          options={'xatol': 1e-12}).x)
        return self

    def predict(self, inputs):
        return np.array([response(e, self.rate) for e in inputs])
