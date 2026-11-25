from functools import lru_cache
import numpy as np
from numpy.polynomial.legendre import leggauss


@lru_cache(256)
def transport(amplitude,force,points=192,quadrature=96):
    period=2*np.pi
    x=period*np.arange(points)/points
    nodes,weights=leggauss(quadrature)
    distance=np.pi*(nodes+1);weights=np.pi*weights
    potential=amplitude*np.cos(x)
    forward=np.exp(potential[:,None]-amplitude*np.cos(x[:,None]-distance)-force*distance)
    integral=forward@weights
    mean=np.mean(integral)
    mean_derivative=-np.mean(forward@(weights*distance))
    numerator=-np.expm1(-force*period)
    numerator_derivative=period*np.exp(-force*period)
    drift=numerator/mean
    differential_mobility=numerator_derivative/mean-numerator*mean_derivative/mean**2
    spread=differential_mobility
    return drift,spread


def predict_at(experiments,friction):
    out=[]
    for e in experiments:
        values=transport(e['amplitude'],e['force'])
        out.append(values[0 if e['observable']=='drift' else 1]/friction)
    return np.array(out)


class Model:
    def __init__(self):
        self.friction=None

    def fit(self,records):
        inputs=[r['input'] for r in records]
        design=predict_at(inputs,1.)
        values=np.array([r['value'] for r in records]);sigma=np.array([r['sigma'] for r in records])
        inverse=np.sum(design*values/sigma**2)/np.sum((design/sigma)**2)
        self.friction=float(np.clip(1/inverse,.5,1.5))
        return self

    def predict(self,experiments):
        return predict_at(experiments,self.friction)
