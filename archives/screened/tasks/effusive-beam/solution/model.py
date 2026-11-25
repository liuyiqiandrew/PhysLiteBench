from scipy.optimize import minimize_scalar
import numpy as np
from scipy.special import ndtr
MASS = 39.948*1.66053906660e-27
KB = 1.380649e-23


class Model:
    def __init__(self):
        self.temperature = None

    def fit(self, records):
        experiments = [r['input'] for r in records]
        values = np.array([r['value'] for r in records])
        sigma = np.array([r['sigma'] for r in records])
        def loss(value):
            self.temperature = value
            return np.sum(((self.predict(experiments)-values)/sigma)**2)
        answer = minimize_scalar(loss, bounds=(150.0, 1000.0), method='bounded', options={'xatol': 1e-11})
        self.temperature = float(answer.x)
        return self

    def predict(self, experiments):
        scale = np.sqrt(KB*self.temperature/MASS)
        return np.array([ndtr(e['threshold']/scale) if e['component'] in ('x', 'y')
                         else -np.expm1(-max(0., e['threshold'])**2/(2*scale**2)) for e in experiments])
