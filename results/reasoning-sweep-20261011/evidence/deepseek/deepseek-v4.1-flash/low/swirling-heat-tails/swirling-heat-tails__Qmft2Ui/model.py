import numpy as np


def response(e):
    r, z = e['drive_ratio'], e['relative_heat']
    q = 2*r*r*z
    rate = np.sqrt((1+r*r)*(q*q+4*r*r)/(4*r*r))-q/2-1
    return e['stiffness']*rate


def predict_at(experiments, mobility):
    return mobility*np.asarray([response(e) for e in experiments])


class Model:
    def __init__(self):
        self.mobility = None

    def fit(self, records):
        raise NotImplementedError('Fit the common mobility from calibration records.')

    def predict(self, experiments):
        return predict_at(experiments, self.mobility)
