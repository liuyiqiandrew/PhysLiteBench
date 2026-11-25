from scipy.optimize import minimize_scalar
import numpy as np
AMOUNT = 1.
ATTRACTION = .36


class Model:
    def __init__(self):
        self.heat_capacity = None

    def fit(self, records):
        experiments = [r['input'] for r in records]
        values = np.array([r['value'] for r in records])
        sigma = np.array([r['sigma'] for r in records])
        def loss(value):
            self.heat_capacity = value
            return np.sum(((self.predict(experiments)-values)/sigma)**2)
        answer = minimize_scalar(loss, bounds=(15.0, 40.0), method='bounded', options={'xatol': 1e-11})
        self.heat_capacity = float(answer.x)
        return self

    def predict(self, experiments):
        out = []
        for e in experiments:
            if e['protocol'] == 'heating':
                out.append(AMOUNT*self.heat_capacity*(e['final_temperature']-e['initial_temperature']))
            else:
                out.append(e['initial_temperature'])
        return np.array(out)
