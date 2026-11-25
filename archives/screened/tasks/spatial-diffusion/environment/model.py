from functools import lru_cache
import numpy as np
from scipy.linalg import eigh_tridiagonal

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
        raise NotImplementedError

    def predict(self, experiments):
        out = []
        for e in experiments:
            initial = np.zeros(MODES)
            initial[:2] = [e['first']/2, e['second']/2]
            rates, vectors = spectrum(e['contrast'])
            decay = np.exp(-self.diffusivity*(2*np.pi/LENGTH)**2*e['time']*rates)
            n = np.arange(1, MODES+1)
            a = e['contrast']
            ratio = -a/(1+np.sqrt(1-a*a))
            stationary = ratio**n
            moments = stationary+n*(vectors @ (decay*(vectors.T @ ((initial-stationary)/n))))
            out.append(moments[e['mode']-1])
        return np.array(out)
