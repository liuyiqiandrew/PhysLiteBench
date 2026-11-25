import numpy as np
from scipy.special import ndtr
MASS = 39.948*1.66053906660e-27
KB = 1.380649e-23


class Model:
    def __init__(self):
        self.temperature = None

    def fit(self, records):
        raise NotImplementedError

    def predict(self, experiments):
        scale = np.sqrt(KB*self.temperature/MASS)
        return np.array([ndtr(e['threshold']/scale) if e['component'] in ('x', 'y')
                         else max(0., 2*ndtr(e['threshold']/scale)-1) for e in experiments])
