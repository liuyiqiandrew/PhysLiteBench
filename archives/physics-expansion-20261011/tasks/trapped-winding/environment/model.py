import numpy as np


def predict_at(experiments, mobility):
    return mobility*np.asarray([e['stiffness']*e['probe']**2 for e in experiments])


class Model:
    def __init__(self):
        self.mobility = None

    def fit(self, records):
        raise NotImplementedError('Fit mobility from the calibration records.')

    def predict(self, experiments):
        return predict_at(experiments, self.mobility)
