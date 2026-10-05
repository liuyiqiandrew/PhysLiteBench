"""Linear rod evolution and conductivity calibration interface."""
from functools import lru_cache

import numpy as np
from scipy.linalg import eig, solve
from scipy.optimize import minimize_scalar

LENGTH = .05
AREA = 1e-4
TEMPERATURE = 300.
HEAT_CAPACITY = 1e6
BODY_CAPACITY = 6.
MODULUS_0 = MODULUS_1 = 2e9
STRAIN_SCALE = .002
CELLS = 80


def material(x, chi):
    beta0 = MODULUS_0 * .002 * (1 + .65*np.cos(2*np.pi*x/LENGTH))
    beta1 = chi*beta0
    viscosity = MODULUS_1*12*(1 + .97*np.cos(2*np.pi*x/LENGTH))
    return beta0, beta1, viscosity


def operators(conductivity, contact, chi, cells=CELLS):
    dx = LENGTH/cells
    x = (np.arange(cells)+.5)*dx
    b0, b1, viscosity = material(x, chi)
    identity = np.eye(cells)
    project = identity - np.ones((cells, cells))/cells
    strain_t = project*(b0+b1)[None, :]/(MODULUS_0+MODULUS_1)
    strain_z = MODULUS_1/(MODULUS_0+MODULUS_1)*project
    z_t = (MODULUS_1*strain_t-np.diag(b1))/viscosity[:, None]
    z_z = (MODULUS_1*strain_z-MODULUS_1*identity)/viscosity[:, None]
    capacity = np.diag(HEAT_CAPACITY+TEMPERATURE*b1**2/MODULUS_1) + TEMPERATURE*b0[:, None]*strain_t
    storage = TEMPERATURE*b0[:, None]*strain_z
    heat = np.zeros((cells+1, cells+1))
    edge = conductivity*AREA/dx
    for j in range(cells-1):
        heat[j, j] += edge
        heat[j+1, j+1] += edge
        heat[j, j+1] -= edge
        heat[j+1, j] -= edge
    h = 0. if contact == 0 else 1/(1/contact+dx/(2*conductivity*AREA))
    heat[0, 0] += h
    heat[-1, -1] = h
    heat[0, -1] = heat[-1, 0] = -h
    rhs = np.zeros((cells, 2*cells+1))
    rhs[:, :cells] = -heat[:cells, :cells]/(AREA*dx)-storage@z_t
    rhs[:, cells:2*cells] = -storage@z_z
    rhs[:, -1] = -heat[:cells, -1]/(AREA*dx)
    generator = np.zeros((2*cells+1, 2*cells+1))
    generator[:cells] = solve(capacity, rhs)
    generator[cells:2*cells, :cells] = z_t
    generator[cells:2*cells, cells:2*cells] = z_z
    generator[-1, :cells] = -heat[-1, :cells]/BODY_CAPACITY
    generator[-1, -1] = -heat[-1, -1]/BODY_CAPACITY
    scale = np.r_[np.ones(cells), np.full(cells, STRAIN_SCALE), 1.]
    generator *= scale[None, :]/scale[:, None]
    return generator, x, b0, b1, strain_t, strain_z, heat


def preparation(chi, cells=CELLS):
    x = (np.arange(cells)+.5)*LENGTH/cells
    b0, b1, _ = material(x, chi)
    theta = np.column_stack([np.ones(cells), np.cos(np.pi*x/LENGTH), np.cos(2*np.pi*x/LENGTH), np.zeros(cells)])
    strain = (b0[:, None]*theta-np.mean(b0[:, None]*theta, axis=0))/MODULUS_0
    z = strain-b1[:, None]*theta/MODULUS_1
    return np.vstack([theta, z/STRAIN_SCALE, [0., 0., 0., 1.]])


@lru_cache(maxsize=192)
def spectrum(conductivity, contact, chi, cells=CELLS):
    generator, x, *_ = operators(conductivity, contact, chi, cells)
    rates, vectors = eig(generator)
    amplitudes = solve(vectors, preparation(chi, cells))
    output = np.zeros((4, 2*cells+1))
    output[0, :cells] = 1/cells
    output[1, :cells] = np.cos(np.pi*x/LENGTH)/cells
    output[2, :cells] = np.cos(2*np.pi*x/LENGTH)/cells
    output[3, -1] = 1.
    coefficients = (output@vectors)[:, :, None]*amplitudes[None, :, :]
    return rates, coefficients


def predict_values(experiments, conductivity, cells=CELLS):
    if not 80. <= conductivity <= 220.:
        raise ValueError('conductivity must be in [80,220]')
    names = {'mean': 0, 'first': 1, 'second': 2, 'bath': 3}
    result = []
    for experiment in experiments:
        contact, chi, time = (float(experiment[k]) for k in ('contact', 'chi', 'time'))
        if not (0 <= contact <= 1.2 and 0 <= chi <= 1 and 0 <= time <= 80):
            raise ValueError('experiment outside the documented range')
        initial = np.array([experiment[k] for k in ('mean', 'first', 'second', 'bath_initial')], dtype=float)
        rates, coefficients = spectrum(float(conductivity), contact, chi, cells)
        result.append(float((np.exp(rates*time)@coefficients[names[experiment['observable']]]@initial).real))
    return np.asarray(result)


class Model:
    def __init__(self, conductivity=None):
        self.conductivity = conductivity

    def fit(self, records):
        inputs = [record['input'] for record in records]
        values = np.array([record['value'] for record in records])
        sigma = np.array([record['sigma'] for record in records])
        def objective(conductivity):
            residual = (predict_values(inputs, conductivity)-values)/sigma
            return float(residual@residual)
        result = minimize_scalar(objective, bounds=(80., 220.), method='bounded',
                                 options={'xatol': 1e-8})
        if not result.success:
            raise RuntimeError('Conductivity fit did not converge.')
        self.conductivity = float(result.x)
        return self

    def predict(self, experiments):
        if self.conductivity is None:
            raise ValueError('Fit conductivity before prediction.')
        return predict_values(experiments, self.conductivity)
