import numpy as np
from scipy.linalg import expm


def flux_jacobian(energy, flux):
    flux = np.asarray(flux, dtype=float)
    magnitude = np.linalg.norm(flux)
    matrix = np.zeros((4, 4))
    matrix[0, 1] = 1.0
    if magnitude == 0.0:
        matrix[1, 0] = 1.0 / 3.0
        return matrix
    f = magnitude / energy
    direction = flux / magnitude
    s = np.sqrt(4.0 - 3.0 * f**2)
    denominator = 5.0 + 2.0 * s
    b = f**2 * (6.0 + 3.0 / (2.0 + s)) / denominator
    a = (1.0 - b) / 3.0
    derivative = (8.0*f*denominator + 6.0*f*(3.0+4.0*f**2)/s) / denominator**2
    tensor = np.outer(direction, direction)
    identity = np.eye(3)
    matrix[1:, 0] = ((a + f*derivative/2)*identity +
                     (b - 3*f*derivative/2)*tensor)[0, :]
    for j in range(3):
        axis = identity[:, j]
        column = derivative*direction[j]*(3*tensor-identity)/2
        column += f*(6+3/(2+s))/denominator * (
            np.outer(axis, direction) + np.outer(direction, axis) - 2*direction[j]*tensor)
        matrix[1:, 1+j] = column[0, :]
    return matrix


def transport(experiment):
    directions = np.asarray(experiment["directions"], dtype=float)
    weights = np.asarray(experiment["weights"], dtype=float)
    amplitudes = weights * np.asarray(experiment["modulations"], dtype=float)
    matrix = flux_jacobian(float(np.sum(weights)), weights @ directions)
    initial = np.r_[np.sum(amplitudes), amplitudes @ directions]
    evolved = expm(-1j * experiment["wavenumber"] * experiment["time"] * matrix) @ initial
    phase = np.exp(1j * experiment["wavenumber"] * experiment["position"])
    return float(np.real(phase * evolved[0]))


class Model:
    def __init__(self):
        self.absorption = .25

    def fit(self, records):
        raise NotImplementedError("Implement calibration fitting.")

    def predict(self, experiments):
        return np.asarray([transport(e) * np.exp(-self.absorption*e["time"])
                           for e in experiments], dtype=float)
