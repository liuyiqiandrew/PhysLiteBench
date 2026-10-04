import numpy as np


class Model:
    def __init__(self):
        self.affinity_a = None
        self.affinity_b = None

    def fit(self, records):
        raise NotImplementedError

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
