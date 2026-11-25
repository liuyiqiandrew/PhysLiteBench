"""Stationary periodic backward-generator and displacement cell-corrector solve."""
from functools import lru_cache
import numpy as np

PARAMETER='friction'
TRUE_PARAMETER=.9


@lru_cache(8)
def derivatives(points):
    x=2*np.pi*np.arange(points)/points
    wave=np.fft.fftfreq(points,1/points)
    transform=np.fft.fft(np.eye(points),axis=0)
    first=np.fft.ifft(1j*wave[:,None]*transform,axis=0).real
    second=np.fft.ifft(-wave[:,None]**2*transform,axis=0).real
    return x,first,second


@lru_cache(256)
def cell(amplitude,force,points=96):
    x,first,second=derivatives(points)
    local_drift=force+amplitude*np.sin(x)
    generator=local_drift[:,None]*first+second
    stationarity=generator.T.copy();stationarity[0]=1.
    rhs=np.zeros(points);rhs[0]=1.
    probability=np.linalg.solve(stationarity,rhs)
    mean_velocity=probability@local_drift
    problem=generator.copy();problem[0]=probability
    rhs=mean_velocity-local_drift;rhs[0]=0.
    corrector=np.linalg.solve(problem,rhs)
    dispersion=probability@(1+first@corrector)**2
    return mean_velocity,dispersion,probability,corrector,generator


def predict(experiments,friction=TRUE_PARAMETER,points=96):
    out=[]
    for e in experiments:
        drift,diffusion,*_=cell(e['amplitude'],e['force'],points)
        out.append((drift if e['observable']=='drift' else diffusion)/friction)
    return np.array(out)


def reading(amplitude,force,observable='diffusion'):
    return dict(amplitude=float(amplitude),force=float(force),observable=observable)


def calibration_inputs():
    return [reading(a,0.) for a in np.linspace(0,3,72)]


def hidden_inputs():
    return {name:[reading(a,f,o) for a in amplitudes for f in forces for o in ['drift','diffusion']]
            for name,amplitudes,forces in [
                ('moderate_landscape',[1.,1.5],[.7,1.,1.5,2.]),
                ('strong_landscape',[2.,3.],[1.,1.5,2.,2.5,3.]),
                ('reversed_drive',[1.2,2.5],[-.7,-1.2,-2.,-3.])]}
