import numpy as np


def predict_at(experiments, mobility):
    return mobility*np.asarray([e['stiffness']*abs(e['probe']) for e in experiments])


class Model:
    def __init__(self):
        self.mobility = None

    def fit(self, records):
        basis = predict_at([r['input'] for r in records],1.)
        values = np.asarray([r['value'] for r in records])
        weights = 1/np.asarray([r['sigma'] for r in records])**2
        self.mobility = float(np.clip(np.dot(weights*basis,values)/
                                      np.dot(weights*basis,basis),.7,1.4))
        return self

    def predict(self, experiments):
        return predict_at(experiments, self.mobility)
