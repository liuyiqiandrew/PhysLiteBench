import numpy as np
from scipy.optimize import minimize_scalar


def polarizabilities(experiments,scale):
    k=np.array([e['k'] for e in experiments])
    ae=scale*np.array([e['electric_weight'] for e in experiments])
    am=scale*np.array([e['magnetic_weight'] for e in experiments])
    return k,ae/(1-1j*k**3*ae/(6*np.pi)),am/(1-1j*k**3*am/(6*np.pi))


class Model:
    def __init__(self):
        self.polarizability_scale=None

    def fit(self,records):
        raise NotImplementedError

    def predict(self,experiments):
        k,electric,magnetic=polarizabilities(experiments,self.polarizability_scale)
        force=k*np.imag(electric+magnetic)/2

        return force
