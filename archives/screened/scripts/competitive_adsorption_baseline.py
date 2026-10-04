import numpy as np
from scipy.optimize import least_squares


class Model:
    def __init__(self):
        self.affinity_a = None
        self.affinity_b = None

    def fit(self, records):
        experiments = [r['input'] for r in records]
        values = np.array([r['value'] for r in records])
        sigma = np.array([r['sigma'] for r in records])
        def residual(parameters):
            self.affinity_a, self.affinity_b = parameters
            return (self.predict(experiments)-values)/sigma
        answer = least_squares(residual, [1., 1.], bounds=(.2, 3.),
                               ftol=1e-12, xtol=1e-12, gtol=1e-12)
        self.affinity_a, self.affinity_b = map(float, answer.x)
        return self

    def predict(self, experiments):
        result = []
        for e in experiments:
            if e['species'] == 'A':
                weight = self.affinity_a*e['concentration_a']
            else:
                weight = self.affinity_b*e['concentration_b']
            occupancy = weight/(1+weight)
            if e['protocol'] == 'displacement':
                result.append(0.)
            else:
                result.append(occupancy)
        return np.array(result)
