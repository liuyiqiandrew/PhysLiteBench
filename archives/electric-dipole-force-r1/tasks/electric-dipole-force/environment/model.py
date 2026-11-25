import numpy as np


def field_data(experiment):
    electric = np.zeros(3, dtype=complex)
    magnetic = np.zeros(3, dtype=complex)
    derivative = np.zeros((3, 3), dtype=complex)
    position = np.array(experiment['position'])
    for wave in experiment['waves']:
        direction = np.array(wave['direction'])
        amplitude = np.array(wave['real'])+1j*np.array(wave['imag'])
        local = amplitude*np.exp(1j*direction@position)
        electric += local
        magnetic += np.cross(direction, local)
        derivative += 1j*np.outer(direction, local)
    return electric, magnetic, derivative


def coefficients(experiment):
    electric, magnetic, derivative = field_data(experiment)
    axis = np.array(experiment['axis'])
    first = .5*np.real(derivative@electric.conj())
    second = .5*np.real(np.cross(electric, magnetic.conj()))
    return np.array([axis@first, axis@second])


def polarizability(response_strength):
    return response_strength/(1-1j*response_strength/(6*np.pi))


def predict_at(experiments, response_strength):
    alpha = polarizability(response_strength)
    return np.array([coefficients(e)@np.array([alpha.real, alpha.imag]) for e in experiments])


class Model:
    def __init__(self):
        self.response_strength = None

    def fit(self, records):
        raise NotImplementedError("Fit the electric response strength from calibration records.")

    def predict(self, experiments):
        return predict_at(experiments, self.response_strength)
