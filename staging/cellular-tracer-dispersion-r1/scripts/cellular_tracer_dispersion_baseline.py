import numpy as np
from scipy.optimize import minimize_scalar


def diffusion_tensor(a,b,c,d,diffusivity):
    first=(a*a+c*c/4)/2
    second=(b*b+d*d/4)/2
    difference=(first-second)/diffusivity
    z=difference-diffusivity
    root=np.sqrt(z*z+4*first)
    yy=2*first/(root+z) if z>0 else (root-z)/2
    xx=diffusivity+first/yy
    return np.diag([xx,yy])


def predict_at(experiments,diffusivity):
    values=[]
    for e in experiments:
        tensor=diffusion_tensor(e['a'],e['b'],e['c'],e['d'],diffusivity)
        direction=np.array([np.cos(e['angle']),np.sin(e['angle'])])
        values.append(direction@tensor@direction)
    return np.asarray(values,dtype=float)


class Model:
    def __init__(self):
        self.diffusivity=None

    def fit(self,records):
        experiments=[r['input'] for r in records]
        values=np.array([r['value'] for r in records])
        sigma=np.array([r['sigma'] for r in records])
        def objective(D):
            residual=(predict_at(experiments,D)-values)/sigma
            return float(residual@residual)
        result=minimize_scalar(objective,bounds=(.8,1.2),method='bounded',options={'xatol':1e-13})
        choices=[.8,float(result.x),1.2]
        self.diffusivity=min(choices,key=objective)
        return self

    def predict(self,experiments):
        return predict_at(experiments,self.diffusivity)
