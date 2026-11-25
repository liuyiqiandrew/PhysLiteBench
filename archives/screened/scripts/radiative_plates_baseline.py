from scipy.optimize import minimize_scalar
import numpy as np
STEFAN_BOLTZMANN = 5.670374419e-8


class Model:
    def __init__(self):
        self.emissivity = None

    def fit(self, records):
        experiments = [r['input'] for r in records]
        values = np.array([r['value'] for r in records])
        sigma = np.array([r['sigma'] for r in records])
        def loss(value):
            self.emissivity = value
            return np.sum(((self.predict(experiments)-values)/sigma)**2)
        answer = minimize_scalar(loss, bounds=(0.05, 0.95), method='bounded', options={'xatol': 1e-11})
        self.emissivity = float(answer.x)
        return self

    def predict(self, experiments):
        return np.array([self.emissivity*e['emissivity_2']*STEFAN_BOLTZMANN*
               (e['temperature_1']**4-e['temperature_2']**4) for e in experiments])
