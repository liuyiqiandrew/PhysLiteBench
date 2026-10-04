from functools import lru_cache
import numpy as np
from scipy.linalg import expm, solve_continuous_lyapunov


@lru_cache(256)
def response(friction, field, temperature, kx, ky, time, cxx, cxy, cyy):
    mobility = np.array([[friction, field], [-field, friction]]) / (friction**2 + field**2)
    drift = -mobility @ np.diag([kx, ky])
    stationary = temperature * np.diag([1 / kx, 1 / ky])
    initial = np.array([[cxx, cxy], [cxy, cyy]])
    difference = initial - stationary
    propagator = expm(time * drift)
    evolved = propagator @ difference @ propagator.T
    covariance = stationary + evolved
    rotation = np.array([[0., 1.], [-1., 0.]])
    area_rate = (rotation @ drift + (rotation @ drift).T) / 2
    integral = solve_continuous_lyapunov(drift.T, -area_rate)
    area = np.trace(integral @ (difference - evolved)) + time * np.trace(area_rate @ stationary)
    return covariance, float(area)


def predict_at(experiments, friction):
    out = []
    indices = {'xx': (0, 0), 'xy': (0, 1), 'yy': (1, 1)}
    for e in experiments:
        c = e['initial_covariance']
        covariance, area = response(float(friction), e['field'], e['temperature'],
                                    *e['stiffness'], e['time'], c[0][0], c[0][1], c[1][1])
        out.append(area if e['readout'] == 'area' else covariance[indices[e['readout']]])
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
        result = minimize_scalar(objective, bounds=(.5, 1.5), method='bounded',
                                 options={'xatol': 1e-12})
        self.friction = float(result.x)
        return self

    def predict(self, experiments):
        return predict_at(experiments, self.friction)
