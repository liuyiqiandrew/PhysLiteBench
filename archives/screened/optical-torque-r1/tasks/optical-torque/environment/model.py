import numpy as np


RATIOS = np.array([1., 1.7, 2.6])


def polarizability(strength, orientation):
    values = strength*RATIOS
    return orientation@np.diag(values/(1-1j*values/(6*np.pi)))@orientation.T


def response(experiment, strength):
    orientation = np.array(experiment['orientation'])
    field = np.array(experiment['field_real'])+1j*np.array(experiment['field_imag'])
    dipole = polarizability(strength, orientation)@field
    if experiment['observable'] == 'force':
        vector = .5*np.imag(np.vdot(field, dipole))*np.array(experiment['direction'])
    else:
        vector = .5*np.real(np.cross(dipole, field.conj()))

    return float(np.array(experiment['axis'])@vector)


def predict_at(experiments, strength):
    return np.array([response(e, strength) for e in experiments])


class Model:
    def __init__(self):
        self.response_strength = None

    def fit(self, records):
        raise NotImplementedError("Fit the common response strength from calibration.")

    def predict(self, experiments):
        return predict_at(experiments, self.response_strength)
