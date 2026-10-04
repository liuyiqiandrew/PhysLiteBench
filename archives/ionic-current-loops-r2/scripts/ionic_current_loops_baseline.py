from functools import lru_cache
import json
import numpy as np

RATIOS=np.array([1.,4.,1.])
CHARGES=np.array([1.,1.,-1.])


@lru_cache(128)
def fields(key,points=257):
    e=json.loads(key);x=np.arange(points)*2*np.pi/points
    c=np.array([e['means'][i]+e['amplitudes'][i]*np.cos(e['waves'][i]*x+e['phases'][i]) for i in [0,1]])
    c=np.vstack((c,c.sum(axis=0)))
    modes=np.fft.fftfreq(points,1/points)
    gradient=np.fft.ifft(1j*modes*np.fft.fft(c,axis=1),axis=1).real
    conductivity=RATIOS@c
    diffusion_charge_gradient=(RATIOS*CHARGES)@gradient
    electric=diffusion_charge_gradient/conductivity
    flux=-RATIOS[:,None]*gradient+(RATIOS*CHARGES)[:,None]*c*electric
    change=-np.fft.ifft(1j*modes*np.fft.fft(flux,axis=1),axis=1).real
    return x,c,electric,flux,change


def predict_at(experiments,diffusivity):
    out=[]
    for e in experiments:
        key=json.dumps({k:e[k] for k in ['means','amplitudes','waves','phases']},sort_keys=True)
        x,_,_,_,change=fields(key)
        weight=np.cos(e['detector']*x+e['detector_phase'])
        out.append(2*diffusivity*np.mean(change[e['species']]*weight))
    return np.asarray(out,dtype=float)


class Model:
    def __init__(self):self.diffusivity=None

    def fit(self,records):
        inputs=[r['input'] for r in records]
        values=np.array([r['value'] for r in records]);sigma=np.array([r['sigma'] for r in records])
        coefficient=predict_at(inputs,1.)
        self.diffusivity=float(np.sum(coefficient*values/sigma**2)/np.sum(coefficient**2/sigma**2))
        return self

    def predict(self,experiments):return predict_at(experiments,self.diffusivity)
