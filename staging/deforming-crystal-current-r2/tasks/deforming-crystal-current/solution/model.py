from functools import lru_cache
import numpy as np
from scipy.optimize import brentq, minimize_scalar

S0 = np.array([.08, .06, .22])
E3 = np.array([0., 0., 1.])


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


def capacitance_geometry(f, drive):
    normal = np.linalg.solve(f.T, E3)
    volume = np.linalg.det(f)
    c = volume*np.dot(normal, normal)
    normal_rate = -np.linalg.solve(f.T, drive.T@normal)
    rate = c*np.trace(np.linalg.solve(f, drive))+2*volume*np.dot(normal, normal_rate)
    return c, rate


def loaded_state(f, stiffness, load):
    s = internal_state(f, stiffness)
    if load is None:
        return s
    c, _ = capacitance_geometry(f, np.zeros((3, 3)))
    r = 1/(c+load)
    h = strain_coupling((f.T@f-np.eye(3))/2)
    for _ in range(40):
        x = s-S0
        gradient = stiffness*x+4*np.dot(x, x)*x-h+r*s[2]*E3
        if np.linalg.norm(gradient, np.inf) < 2e-14:
            return s
        hessian = (stiffness+4*np.dot(x, x))*np.eye(3)+8*np.outer(x, x)+r*np.outer(E3, E3)
        s -= np.linalg.solve(hessian, gradient)
    raise RuntimeError('Internal equilibrium did not converge')


@lru_cache(maxsize=16384)
def response(stiffness, f_values, g_values, load):
    f = np.asarray(f_values).reshape(3, 3)
    g = np.asarray(g_values).reshape(3, 3)
    s = loaded_state(f, stiffness, load)
    x = s-S0
    tangent = (stiffness+4*np.dot(x, x))*np.eye(3)+8*np.outer(x, x)
    rhs = strain_coupling((f.T@g+g.T@f)/2)
    if load is None:
        return float(np.linalg.solve(tangent, rhs)[2])
    c, rate = capacitance_geometry(f, g)
    tangent += np.outer(E3, E3)/(c+load)
    rhs += rate*s[2]*E3/(c+load)**2
    ds = np.linalg.solve(tangent, rhs)
    fraction = load/(c+load)
    fraction_rate = -load*rate/(c+load)**2
    return float(fraction*ds[2]+fraction_rate*s[2])


def predict_at(experiments, stiffness):
    return np.asarray([response(float(stiffness),
                                tuple(np.asarray(e['deformation'],dtype=float).ravel()),
                                tuple(np.asarray(e['drive'],dtype=float).ravel()),
                                None if e.get('load') is None else float(e['load']))
                       for e in experiments])


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
        self.stiffness = float(min((result.x, .8, 1.2), key=objective))
        return self

    def predict(self, experiments):
        return predict_at(experiments, self.stiffness)
