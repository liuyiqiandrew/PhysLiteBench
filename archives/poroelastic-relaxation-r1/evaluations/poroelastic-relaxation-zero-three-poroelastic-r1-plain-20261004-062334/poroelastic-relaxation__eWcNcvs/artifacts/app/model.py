import numpy as np
from scipy.optimize import minimize_scalar

ELASTICITY = 1e6 * np.array([[6., 2., .8], [2., 4., .6], [.8, .6, 3.]])
PERMEABILITY = 1e-15 * np.array([[1., .2], [.2, .6]])
TRACE = np.array([1., 1., 0.])
ALPHA = .9
BULK = 20e6
LENGTH = .01
LEAKAGE = 1e-10


def coefficients(e):
    """Return the storage and hydraulic conductance of one Fourier mode.

    The pressure and displacement response of a nonzero cosine mode can be
    written as ``p = P*cos(k.x)`` and ``u = U*sin(k.x)``.  If ``B`` maps the
    displacement amplitude to engineering strain, mechanical equilibrium is

        B.T @ C @ B @ U = alpha * P * k.

    Consequently, the fluid content amplitude is ``S(k)*P`` with the
    wave-vector-dependent storage used below.  The zero mode is special: its
    affine strain is free to adjust, so its storage is the usual zero-mean
    stress (undrained) storage.
    """
    k = 2 * np.pi / LENGTH * np.asarray(e['mode'], dtype=float)
    uniform_strain = np.linalg.solve(ELASTICITY, ALPHA * TRACE)
    uniform_storage = 1 / BULK + ALPHA * TRACE @ uniform_strain

    if np.all(k == 0):
        storage = uniform_storage
    else:
        # e = B @ U for u = U*sin(k.x), using engineering shear strain.
        B = np.array([[k[0], 0.], [0., k[1]], [k[1], k[0]]])
        stiffness = B.T @ ELASTICITY @ B
        storage = 1 / BULK + ALPHA**2 * (k @ np.linalg.solve(stiffness, k))

    conductance = k @ PERMEABILITY @ k + LEAKAGE
    return float(storage), float(conductance)


def predict_at(experiments, viscosity):
    result = []
    for e in experiments:
        storage, conductance = coefficients(e)
        result.append(e['initial_pressure'] * np.exp(-conductance * e['time'] / (viscosity * storage)))
    return np.asarray(result)


class Model:
    def __init__(self):
        self.viscosity = None

    def fit(self, records):
        records = list(records)
        experiments = [r['input'] for r in records]
        y = np.asarray([r['value'] for r in records])
        sigma = np.asarray([r['sigma'] for r in records])
        decay = np.array([coefficients(e)[1] * e['time'] / coefficients(e)[0] for e in experiments])
        initial = np.array([e['initial_pressure'] for e in experiments])
        def loss(viscosity):
            residual = (initial * np.exp(-decay / viscosity) - y) / sigma
            return float(residual @ residual)
        result = minimize_scalar(loss, bounds=(.0008, .0015), method='bounded',
                                 options={'xatol': 1e-14})
        self.viscosity = float(result.x)
        return self

    def predict(self, experiments):
        return predict_at(experiments, self.viscosity)
