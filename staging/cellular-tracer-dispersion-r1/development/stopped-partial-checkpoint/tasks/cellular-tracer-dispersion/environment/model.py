import numpy as np


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
        raise NotImplementedError

    def predict(self,experiments):
        return predict_at(experiments,self.diffusivity)
