from functools import lru_cache
import numpy as np
from scipy.optimize import minimize_scalar

DAMPING = .04
ROTATION = np.kron(np.eye(2), np.array([[0., -1.], [1., 0.]]))


def energy_matrix(kx, ky, gap):
    k = np.hypot(kx, ky)
    direction = ky/k
    p = -np.expm1(-k)/k
    q = np.exp(-k*gap)*(-np.expm1(-k))**2/(2*k)
    matrix = np.zeros((4, 4), dtype=complex)
    matrix[np.ix_([0, 2], [0, 2])] = direction**2*np.array([[1-p, q], [q, 1-p]])
    matrix[np.ix_([1, 3], [1, 3])] = np.array([[p, -q], [-q, p]])
    matrix[0, 3] = matrix[1, 2] = -1j*direction*q
    matrix[3, 0] = matrix[2, 1] = 1j*direction*q
    matrix += np.diag(np.repeat([.45, .70], 2)+.08*k*k)
    return matrix


@lru_cache(maxsize=512)
def mode_basis(kx, ky, gap):
    energy = energy_matrix(kx, ky, gap)
    generator = (ROTATION-DAMPING*np.eye(4))@energy/(1+DAMPING**2)
    rates, vectors = np.linalg.eig(generator)
    return rates, vectors, np.linalg.inv(vectors)


def predict_at(experiments, gyro_rate):
    result = []
    for experiment in experiments:
        kx, ky = experiment['wavevector']
        rates, vectors, inverse = mode_basis(kx, ky, experiment['gap'])
        initial = np.zeros(4, dtype=complex)
        initial[[0, 2]] = np.array(experiment['initial_real'])+1j*np.array(experiment['initial_imag'])
        state = vectors@(np.exp(rates*gyro_rate*experiment['time'])*(inverse@initial))
        value = state[2*experiment['layer']+1]
        result.append(value.real if experiment['quadrature']=='real' else value.imag)
    return np.array(result)


class Model:
    def __init__(self):
        self.gyro_rate = None

    def fit(self, records):
        inputs = [r['input'] for r in records]
        values = np.array([r['value'] for r in records])
        sigma = np.array([r['sigma'] for r in records])
        result = minimize_scalar(lambda g: np.sum(((predict_at(inputs, g)-values)/sigma)**2),
                                 bounds=(.7, 1.3), method='bounded', options={'xatol': 1e-12})
        self.gyro_rate = float(result.x)
        return self

    def predict(self, experiments):
        return predict_at(experiments, self.gyro_rate)
