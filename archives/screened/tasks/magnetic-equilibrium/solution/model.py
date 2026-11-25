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
            sign = np.sign(h)
            m = brentq(lambda z: z-np.tanh((self.coupling*z+abs(h))/t), 0., 1., xtol=1e-13)
            result.append(sign*m)
        return np.array(result)
