import numpy as np

CHARGE = 1.602176634e-19


class Model:
    def __init__(self):
        self.mobility = None

    def fit(self, records):
        raise NotImplementedError

    def predict(self, experiments):
        mu = self.mobility
        result = []
        for e in experiments:
            denominator = 1+(mu*e['magnetic_field'])**2
            positive = CHARGE*e['density_positive']*mu/denominator
            negative = CHARGE*e['density_negative']*mu/denominator
            result.append((positive+negative)*e['electric_field'])
        return np.array(result)
