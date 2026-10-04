from functools import lru_cache
import numpy as np
from scipy.linalg import eigh_tridiagonal
from scipy.optimize import minimize_scalar

LENGTH = 10.
MODES = 36


@lru_cache(None)
def spectrum(contrast):
    n = np.arange(1, MODES+1, dtype=float)
    return eigh_tridiagonal(n*n, contrast*n[:-1]*n[1:]/2)


class Model:
    def __init__(self):
        self.diffusivity = None

    def fit(self, records):
        experiments = [r['input'] for r in records]
        values = np.array([r['value'] for r in records])
        sigma = np.array([r['sigma'] for r in records])
        def loss(value):
            self.diffusivity = value
            return np.sum(((self.predict(experiments)-values)/sigma)**2)
        answer = minimize_scalar(loss, bounds=(.1, 1.2), method='bounded', options={'xatol': 1e-11})
        self.diffusivity = float(answer.x)
        return self

    def predict(self, experiments):
        out = []
        for e in experiments:
            initial = np.zeros(MODES)
            initial[:2] = [e['first']/2, e['second']/2]
            rates, vectors = spectrum(e['contrast'])
            decay = np.exp(-self.diffusivity*(2*np.pi/LENGTH)**2*e['time']*rates)
            moments = vectors @ (decay*(vectors.T @ initial))
            out.append(moments[e['mode']-1])
        return np.array(out)
