from functools import lru_cache
import numpy as np
from scipy.optimize import brentq, minimize_scalar

S0 = np.array([.08, .06, .22])


def strain_coupling(e):
    return np.array([.60*e[0, 2]+.20*e[0, 1],
                     .50*e[1, 2]-.10*e[0, 1],
                     .45*e[0, 0]+.30*e[1, 1]+.70*e[2, 2]+.20*e[0, 1]])


def internal_state(f, stiffness):
    h = strain_coupling((f.T@f-np.eye(3))/2)
    norm = np.linalg.norm(h)
    if norm == 0:
        return S0.copy()
    radius = brentq(lambda r: stiffness*r+4*r**3-norm, 0, norm/stiffness,
                    xtol=1e-15)
    return S0+radius*h/norm


@lru_cache(maxsize=16384)
def response(stiffness, f_values, g_values):
    f = np.asarray(f_values).reshape(3, 3)
    g = np.asarray(g_values).reshape(3, 3)
    s = internal_state(f, stiffness)
    x = s-S0
    tangent = (stiffness+4*np.dot(x, x))*np.eye(3)+8*np.outer(x, x)
    ds = np.linalg.solve(tangent, strain_coupling((f.T@g+g.T@f)/2))
    volume = np.linalg.det(f)
    polarization_rate = -(g@s+f@ds)/volume
    polarization_rate += f@s*np.trace(np.linalg.solve(f, g))/volume
    face = volume*np.linalg.solve(f.T, np.array([0., 0., 1.]))
    return float(-face@polarization_rate)


def predict_at(experiments, stiffness):
    return np.asarray([response(float(stiffness),tuple(np.asarray(e['deformation'],dtype=float).ravel()),
                                tuple(np.asarray(e['drive'],dtype=float).ravel())) for e in experiments])


class Model:
    def __init__(self):
        self.stiffness = None

    def fit(self, records):
        inputs = [row['input'] for row in records]
        values = np.asarray([row['value'] for row in records])
        sigma = np.asarray([row['sigma'] for row in records])
        def objective(k):
            return float(np.mean(((predict_at(inputs, k)-values)/sigma)**2))
        result = minimize_scalar(objective, bounds=(.8, 1.2), method='bounded',
                                 options={'xatol': 1e-13})
        self.stiffness = min((result.x, .8, 1.2), key=objective)
        return self

    def predict(self, experiments):
        return predict_at(experiments, self.stiffness)
