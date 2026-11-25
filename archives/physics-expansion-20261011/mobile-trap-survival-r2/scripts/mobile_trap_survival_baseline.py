import numpy as np


def response(e):
    return e['density']*abs(e['force'])


def predict_at(experiments, mobility):
    return mobility*np.asarray([response(e) for e in experiments])


class Model:
    def __init__(self):
        self.mobility = None

    def fit(self, records):
        basis = np.asarray([response(r['input']) for r in records])
        values = np.asarray([r['value'] for r in records])
        weights = 1/np.asarray([r['sigma'] for r in records])**2
        self.mobility = float(np.sum(weights*basis*values)/np.sum(weights*basis*basis))
        return self

    def predict(self, experiments):
        return predict_at(experiments, self.mobility)
