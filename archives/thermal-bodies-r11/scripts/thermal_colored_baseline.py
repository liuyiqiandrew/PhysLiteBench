from functools import lru_cache
import numpy as np
from scipy.linalg import expm, solve_continuous_lyapunov


def stiffness(springs, coupling):
    return np.diag(springs) + coupling * np.array([[1., -1.], [-1., 1.]])

@lru_cache(512)
def stationary_modes(friction, k1, k2, coupling, memory, t1, t2):
    eigenvalues, vectors = np.linalg.eigh(stiffness([k1, k2], coupling))
    modes = []
    for alpha in range(2):
        rates = friction * vectors[:, alpha]**2
        drift = np.array([[0., 1., 0., 0.],
                          [-eigenvalues[alpha], 0., 1., 1.],
                          [0., -rates[0]/memory, -1/memory, 0.],
                          [0., -rates[1]/memory, 0., -1/memory]])
        diffusion = np.diag([0., 0., 2*rates[0]*t1/memory**2,
                            2*rates[1]*t2/memory**2])
        covariance = solve_continuous_lyapunov(drift, -diffusion)
        modes.append((drift, covariance))
    return modes


def predict_at(experiments, friction):
    out = []
    for e in experiments:
        modes = stationary_modes(float(friction), *e['springs'], e['coupling'],
                                 e['memory'], *e['temperatures'])
        if e['readout'] == 'mode_correlation':
            drift, covariance = modes[e['mode']]
            out.append((expm(e['lag'] * drift) @ covariance)[1, 1])
        else:
            out.append(sum(covariance[1, 2 + e['bath']] for _, covariance in modes))
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
