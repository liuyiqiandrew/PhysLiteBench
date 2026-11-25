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
        result = []
        mu = self.mobility
        for e in experiments:
            total = e['density_positive']+e['density_negative']
            imbalance = (e['density_positive']-e['density_negative'])/total
            field = mu*e['magnetic_field']
            conductivity = CHARGE*total*mu*(1+(imbalance*field)**2)/(1+field**2)
            result.append(conductivity*e['electric_field'])
        return np.array(result)
