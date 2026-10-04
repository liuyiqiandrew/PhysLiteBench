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
    """Return the pressure storage and hydraulic conductance of a mode.

    For a pressure Fourier mode, write the displacement Fourier amplitude as
    ``v`` (the harmless factor ``1j`` has been absorbed into ``v``).  If
    ``n`` is the integer mode vector, its engineering strain is ``B @ v``
    with the matrix below.  Mechanical equilibrium is

        B.T @ (C @ B @ v - alpha * p * TRACE) = 0.

    Consequently, the fluid content is ``storage * p``.  The zero mode is
    special: its affine strain is free to adjust, so all mean stresses vanish
    directly rather than being constrained by a wave-vector equilibrium
    equation.
    """
    mode = np.asarray(e['mode'], dtype=float)
    k = 2 * np.pi / LENGTH * mode

    if np.all(mode == 0):
        # sigma = 0 gives e = C^{-1} alpha*p*TRACE.
        compliance_strain = np.linalg.solve(ELASTICITY, ALPHA * TRACE)
        storage = 1 / BULK + ALPHA * TRACE @ compliance_strain
    else:
        # The common magnitude of the wave vector cancels from this
        # mechanical calculation, so using the integer mode keeps the
        # linear solve better scaled.
        B = np.array([[mode[0], 0.0],
                      [0.0, mode[1]],
                      [mode[1], mode[0]]])
        equilibrium = B.T @ ELASTICITY @ B
        pressure_strain = B @ np.linalg.solve(equilibrium, B.T @ TRACE)
        storage = 1 / BULK + ALPHA ** 2 * TRACE @ pressure_strain

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
