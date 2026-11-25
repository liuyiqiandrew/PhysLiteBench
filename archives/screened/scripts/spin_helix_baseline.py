import numpy as np
from scipy.optimize import minimize_scalar
from scipy.linalg import expm

SPIN_ORBIT = 1.


class Model:
    def __init__(self):
        self.diffusivity = None

    def fit(self,records):
        times=np.array([r['input']['time'] for r in records])
        amplitude=np.array([r['input']['cosine'][r['input']['component']] for r in records])
        values=np.array([r['value'] for r in records])
        sigma=np.array([r['sigma'] for r in records])
        def loss(d):
            return np.sum(((amplitude*np.exp(-d*SPIN_ORBIT**2*times)-values)/sigma)**2)
        result=minimize_scalar(loss,bounds=(.04,.3),method='bounded',options={'xatol':1e-13})
        self.diffusivity=float(result.x)
        return self

    def predict(self,experiments):
        out=[]
        for e in experiments:
            mode=e['mode']
            initial=np.array(e['cosine'])-1j*np.array(e['sine'])
            diffusion=self.diffusivity*mode**2*np.eye(3)
            relaxation=self.diffusivity*SPIN_ORBIT**2*np.diag([1.,0.,1.])
            state=expm(-e['time']*(diffusion+relaxation))@initial
            component=state[e['component']]
            out.append(float(component.real if e['quadrature']=='cosine' else -component.imag))
        return np.array(out)
