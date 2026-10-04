from functools import lru_cache
import numpy as np
from scipy.linalg import expm, solve_continuous_lyapunov


def stiffness(springs, coupling):
    return np.diag(springs) + coupling * np.array([[1., -1.], [-1., 1.]])

@lru_cache(512)
def stationary_state(friction, k1, k2, coupling, memory, t1, t2):
    spring = stiffness([k1, k2], coupling)
    drift = np.zeros((6, 6))
    drift[:2, 2:4] = np.eye(2)
    drift[2:4, :2] = -spring
    drift[2:4, 4:] = np.eye(2)
    drift[4:, 2:4] = -friction / memory * np.eye(2)
    drift[4:, 4:] = -np.eye(2) / memory
    diffusion = np.zeros((6, 6))
    diffusion[4:, 4:] = np.diag([2*friction*t1/memory**2, 2*friction*t2/memory**2])
    covariance = solve_continuous_lyapunov(drift, -diffusion)
    return drift, covariance, np.linalg.eigh(spring)[1]


def predict_at(experiments, friction):
    out = []
    for e in experiments:
        drift, covariance, vectors = stationary_state(float(friction), *e['springs'],
                            e['coupling'], e['memory'], *e['temperatures'])
        if e['readout'] == 'mode_correlation':
            vector = vectors[:, e['mode']]
            temporal = (expm(e['lag'] * drift) @ covariance)[2:4, 2:4]
            out.append(vector @ temporal @ vector)
        else:
            i = e['bath']
            out.append(covariance[2+i, 4+i])
    return np.array(out)


class Model:
    def __init__(self):
        self.friction = None

    def fit(self, records):
        from scipy.optimize import minimize_scalar
        inputs = [r['input'] for r in records]
        values = np.array([r['value'] for r in records])
        sigma = np.array([r['sigma'] for r in records])
        def objective(friction):
            residual = (predict_at(inputs, friction) - values) / sigma
            return float(residual @ residual)
        result = minimize_scalar(objective, bounds=(.3, 1.2), method='bounded',
                                 options={'xatol': 1e-12})
        self.friction = float(result.x)
        return self

    def predict(self, experiments):
        return predict_at(experiments, self.friction)
