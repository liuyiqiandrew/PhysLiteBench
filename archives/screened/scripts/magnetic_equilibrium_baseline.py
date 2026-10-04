from scipy.optimize import minimize_scalar
import numpy as np
from scipy.optimize import root, brentq



class Model:
    def __init__(self):
        self.coupling = None

    def fit(self, records):
        experiments = [r['input'] for r in records]
        values = np.array([r['value'] for r in records])
        sigma = np.array([r['sigma'] for r in records])
        def loss(value):
            self.coupling = value
            return np.sum(((self.predict(experiments)-values)/sigma)**2)
        answer = minimize_scalar(loss, bounds=(0.3, 1.2), method='bounded', options={'xatol': 1e-11})
        self.coupling = float(answer.x)
        return self

    def predict(self, experiments):
        result = []
        for e in experiments:
            t, h = e['temperature'], e['field']
            answer = root(lambda m: m-np.tanh((self.coupling*m+h)/t), [0.], tol=1e-10)
            result.append(float(answer.x[0]))
        return np.array(result)
