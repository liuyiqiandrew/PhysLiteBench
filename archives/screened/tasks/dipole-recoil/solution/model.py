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
        inputs=[r['input'] for r in records]
        observed=np.array([r['value'] for r in records]);sigma=np.array([r['sigma'] for r in records])
        def objective(scale):
            self.polarizability_scale=scale
            return float(np.sum(((self.predict(inputs)-observed)/sigma)**2))
        self.polarizability_scale=float(minimize_scalar(objective,bounds=(1.,3.),method='bounded',options={'xatol':1e-12}).x)
        return self

    def predict(self,experiments):
        k,electric,magnetic=polarizabilities(experiments,self.polarizability_scale)
        force=k*np.imag(electric+magnetic)/2
        force-=k**4*np.real(electric*magnetic.conj())/(12*np.pi)
        return force
