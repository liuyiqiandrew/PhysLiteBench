import numpy as np
from scipy.linalg import eigh
from scipy.special import roots_legendre


def gap(experiment, modes=80, points=320):
    q, weights = roots_legendre(points)
    x, weights = (q+1)/2, weights/2
    n = np.arange(1, modes+1)
    basis = np.sqrt(2)*np.sin(np.pi*x[:, None]*n)
    potential = experiment['well']*np.cos(2*np.pi*x)
    matrix = np.diag((np.pi*n)**2)+basis.T@((weights*potential)[:, None]*basis)
    values = eigh(matrix, eigvals_only=True, subset_by_index=(0, experiment['level']))
    return float(values[-1]-values[0])


def predict_at(experiments, gain):
    return gain*np.asarray([gap(e) for e in experiments])


class Model:
    def __init__(self):
        self.gain = None

    def fit(self, records):
        basis = predict_at([r['input'] for r in records], 1.)
        sigma = np.asarray([r['sigma'] for r in records])
        values = np.asarray([r['value'] for r in records])
        self.gain = float(np.sum(basis*values/sigma**2)/np.sum((basis/sigma)**2))
        return self

    def predict(self, experiments):
        return predict_at(experiments, self.gain)
