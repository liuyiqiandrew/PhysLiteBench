import numpy as np
from scipy.optimize import minimize_scalar


def energy_levels(experiments,inertia):
    n=np.arange(-24,25,dtype=float)[None,:]
    winding=np.array([e['winding'] for e in experiments])[:,None]
    tilt=np.array([e['tilt'] for e in experiments])[:,None]
    scalar=winding**2*np.sin(tilt)**2/(8*inertia)
    return n**2/(2*inertia)+scalar


def predict_at(experiments,inertia):
    if not experiments:
        return np.empty(0,dtype=float)
    levels=energy_levels(experiments,inertia)
    temperature=np.array([e['temperature'] for e in experiments])[:,None]
    weights=np.exp(-(levels-levels.min(axis=1)[:,None])/temperature)
    weights/=weights.sum(axis=1)[:,None]
    return np.sum(weights*levels,axis=1)


class Model:
    def __init__(self):
        self.inertia=None

    def fit(self,records):
        inputs=[r['input'] for r in records]
        values=np.array([r['value'] for r in records])
        sigma=np.array([r['sigma'] for r in records])
        def objective(inertia):
            residual=(predict_at(inputs,inertia)-values)/sigma
            return float(residual@residual)
        optimum=minimize_scalar(objective,bounds=(.8,1.2),method='bounded',options={'xatol':1e-13})
        self.inertia=float(min([.8,1.2,optimum.x],key=objective))
        return self

    def predict(self,experiments):
        return predict_at(experiments,self.inertia)
