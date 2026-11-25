import numpy as np
from scipy.optimize import minimize_scalar

CHARGE = 1.602176634e-19


class Model:
    def __init__(self):
        self.mobility = None

    def fit(self, records):
        experiments = [r['input'] for r in records]
        values = np.array([r['value'] for r in records])
        sigma = np.array([r['sigma'] for r in records])
        def loss(mobility):
            self.mobility = mobility
            return np.sum(((self.predict(experiments)-values)/sigma)**2)
        answer = minimize_scalar(loss, bounds=(.08, 1.4), method='bounded', options={'xatol': 1e-12})
        self.mobility = float(answer.x)
        return self

    def predict(self, experiments):
        mu = self.mobility
        result = []
        for e in experiments:
            denominator = 1+(mu*e['magnetic_field'])**2
            positive = CHARGE*e['density_positive']*mu/denominator
            negative = CHARGE*e['density_negative']*mu/denominator
            result.append((positive+negative)*e['electric_field'])
        return np.array(result)
