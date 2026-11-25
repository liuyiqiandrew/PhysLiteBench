"""Conservative face fluxes with a periodic scalar-potential solve."""
from functools import lru_cache
import json
import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

TRUE_PARAMETER=1.13


@lru_cache(128)
def fields(key,points):
    e=json.loads(key);h=2*np.pi/points;x=np.arange(points)*h
    c=np.array([e['means'][i]+e['amplitudes'][i]*np.cos(e['waves'][i]*x+e['phases'][i]) for i in [0,1]])
    c=np.vstack((c,c.sum(axis=0)));d=np.array([1.,4.,1.]);z=np.array([1.,1.,-1.])
    conductivity=d@c;g=(d*z)@c
    conductance=(conductivity+np.roll(conductivity,-1))/2/h**2
    diagonal=conductance+np.roll(conductance,1)
    # Fix phi[0]=0. The omitted charge-balance equation follows by summation.
    matrix=diags([-conductance[1:-1],diagonal[1:],-conductance[1:-1]],[-1,0,1],format='csc')
    rhs=(np.roll(g,1)-2*g+np.roll(g,-1))/h**2
    potential=np.zeros(points);potential[1:]=spsolve(matrix,rhs[1:])
    electric=-(np.roll(potential,-1)-potential)/h
    face=(c+np.roll(c,-1,axis=1))/2
    flux=-d[:,None]*(np.roll(c,-1,axis=1)-c)/h+(d*z)[:,None]*face*electric
    change=-(flux-np.roll(flux,1,axis=1))/h
    return x,c,electric,flux,change


def predict(experiments,diffusivity=TRUE_PARAMETER,points=512):
    out=[]
    for e in experiments:
        key=json.dumps({k:e[k] for k in ['means','amplitudes','waves','phases']},sort_keys=True);values=[]
        for n in [points,2*points]:
            x,_,_,_,rate=fields(key,n);weight=np.cos(e['detector']*x+e['detector_phase'])
            values.append(2*np.mean(rate[e['species']]*weight))
        out.append(diffusivity*(4*values[1]-values[0])/3)
    return np.asarray(out)


def experiment(means=(1.,.8),amplitudes=(.8,.64),waves=(1,1),phases=(0.,1.3),species=0,detector=2,detector_phase=np.pi/2):
    return dict(means=list(means),amplitudes=list(amplitudes),waves=list(waves),phases=list(phases),species=species,detector=detector,detector_phase=float(detector_phase))


def calibration_inputs():
    unique=[experiment(means=(.8+.03*i,1.2-.025*i),amplitudes=(.22+.009*i,.18+.013*i),waves=(1,1+i%2),phases=(0.,0.),species=i%3,detector=1+i%2,detector_phase=0.) for i in range(12)]
    return [dict(e) for _ in range(24) for e in unique]


def hidden_inputs():
    out={'phase_quadrature':[],'cosine_harmonics':[],'reversed_profiles':[],'aligned_profiles':[]}
    for i in range(12):
        wave=1+i%2;species=i%3;phase=1.2+.018*i
        out['phase_quadrature'].append(experiment(phases=(0.,phase),waves=(wave,wave),species=species,detector=2*wave))
        out['cosine_harmonics'].append(experiment(phases=(0.,1.72+.012*i),waves=(wave,wave),species=species,detector=2*wave,detector_phase=0.))
        out['reversed_profiles'].append(experiment(phases=(0.,-phase),waves=(wave,wave),species=species,detector=2*wave,detector_phase=-np.pi/2))
        out['aligned_profiles'].append(experiment(phases=(.2,.2),waves=(1,1),species=species,detector=1,detector_phase=.2))
    return out
