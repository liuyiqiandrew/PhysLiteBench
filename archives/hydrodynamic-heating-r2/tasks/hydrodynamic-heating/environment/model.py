import numpy as np
from numpy.polynomial.legendre import leggauss

GAMMA = .06
VISCOSITY = .04
NODES, WEIGHTS = leggauss(64)


def mode_matrix(z, frequency, thickness, angle, plasma_frequency):
    w = frequency
    k = w*np.sin(angle)
    roots = np.roots([VISCOSITY, GAMMA-1j*w-VISCOSITY*w*w,
                      -w*w*(GAMMA-1j*w)-1j*w*plasma_frequency**2])
    vertical = np.sqrt(roots-k*k+0j)
    z = np.atleast_1d(z)
    matrix = np.zeros((len(z), 4, 4), dtype=complex)
    for j in range(2):
        current = (roots[j]-w*w)/(1j*w)
        for side in range(2):
            sign = 1 if side == 0 else -1
            anchor = 0 if side == 0 else thickness
            derivative = 1j*sign*vertical[j]
            wave = np.exp(derivative*(z-anchor))
            matrix[:, :, 2*j+side] = wave[:, None]*np.array(
                [1, derivative, current, derivative*current])
    return matrix


def amplitudes(frequency, thickness, angle, plasma_frequency):
    left, right = mode_matrix([0, thickness], frequency, thickness, angle, plasma_frequency)
    normal = frequency*np.cos(angle)
    boundary = np.array([left[1]+1j*normal*left[0],
                         right[1]-1j*normal*right[0], left[2], right[2]])
    return np.linalg.solve(boundary, [2j*normal, 0, 0, 0])


def response(experiment, plasma_frequency):
    w, d, angle = (experiment[key] for key in ('frequency', 'thickness', 'angle'))
    lower, upper = np.asarray(experiment['window'])*d
    z = (lower+upper)/2+(upper-lower)*NODES/2
    coefficients = amplitudes(w, d, angle, plasma_frequency)
    electric, electric_z, current, current_z = (
        mode_matrix(z, w, d, angle, plasma_frequency) @ coefficients).T
    k = w*np.sin(angle)
    current_zz = ((GAMMA-1j*w+VISCOSITY*k*k)*current-plasma_frequency**2*electric)/VISCOSITY
    heat = GAMMA*abs(current)**2-VISCOSITY*np.real(
        np.conj(current)*(current_zz-k*k*current))
    return float((upper-lower)/2*np.dot(WEIGHTS, heat)/(plasma_frequency**2*np.cos(angle)))


class Model:
    def __init__(self):
        self.plasma_frequency = 1.0

    def fit(self, records):
        raise NotImplementedError("Implement calibration fitting.")

    def predict(self, experiments):
        return np.asarray([response(e, self.plasma_frequency) for e in experiments], dtype=float)
