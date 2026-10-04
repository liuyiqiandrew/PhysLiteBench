from scipy.optimize import minimize_scalar
import numpy as np
DENSITY = 2e21
CHARGE = 1.602176634e-19


class Model:
    def __init__(self):
        self.mobility = None

    def fit(self, records):
        experiments = [r['input'] for r in records]
        values = np.array([r['value'] for r in records])
        sigma = np.array([r['sigma'] for r in records])
        def loss(value):
            self.mobility = value
            return np.sum(((self.predict(experiments)-values)/sigma)**2)
        answer = minimize_scalar(loss, bounds=(0.05, 1.5), method='bounded', options={'xatol': 1e-11})
        self.mobility = float(answer.x)
        return self

    def predict(self, experiments):
        mu = self.mobility
        return np.array([DENSITY*CHARGE*mu*e['electric_field'] /
                         (1+(mu*e['magnetic_field'])**2) for e in experiments])
