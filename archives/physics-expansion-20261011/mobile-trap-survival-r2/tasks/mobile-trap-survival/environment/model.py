import numpy as np


def response(e):
    return e['density']*abs(e['force'])


def predict_at(experiments, mobility):
    return mobility*np.asarray([response(e) for e in experiments])


class Model:
    def __init__(self):
        self.mobility = None

    def fit(self, records):
        raise NotImplementedError('Fit the common target mobility from calibration records.')

    def predict(self, experiments):
        return predict_at(experiments, self.mobility)
