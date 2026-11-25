import numpy as np
from scipy.optimize import brentq

MU = 1e6


def geometry(e):
    s = e['stretch']
    pressure = e['pressure'] / MU
    t = brentq(lambda t: t*t - 1 + 2*np.log(s*t) + pressure*s*t, .2, 2.)
    return s, t


def surface_matrix(x, s, t):
    jacobian = s*t
    c = 1 - 2*np.log(jacobian)
    b = t*t
    longitudinal = np.sqrt(1 - b*x/(b+2+c))
    transverse = np.sqrt(1-x)
    decay = np.array([longitudinal, transverse])
    tangent = np.array([1., transverse])
    normal = np.array([-longitudinal, -1.])
    shear_traction = -b*decay*tangent + c*normal
    normal_traction = -(b+2+c)*decay*normal - 2*tangent
    return np.array([shear_traction, normal_traction])


def mode(e):
    s, t = geometry(e)
    def secular(x):
        matrix = surface_matrix(x, s, t)
        return np.linalg.det(matrix) / (t**4*x)
    x = brentq(secular, 1e-6, 1-1e-10, xtol=1e-13)
    return s*s - t*t + t*t*x


def predict_at(experiments, density):
    return np.sqrt(MU*np.array([mode(e) for e in experiments])/density)


class Model:
    def __init__(self):
        self.density = None

    def fit(self, records):
        records = list(records)
        basis = predict_at([r['input'] for r in records], 1000.)
        values = np.array([r['value'] for r in records])
        sigma = np.array([r['sigma'] for r in records])
        scale = np.sum(basis*values/sigma**2) / np.sum(basis**2/sigma**2)
        self.density = float(1000 / scale**2)
        return self

    def predict(self, experiments):
        return predict_at(experiments, self.density)
