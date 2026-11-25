"""Conservative finite-volume ionic currents with a scalar periodic potential."""
from functools import lru_cache
import json
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

PARAMETER='diffusivity'
TRUE_PARAMETER=1.3


def experiment(means=(1.,1.),amplitudes=(.65,.7),waves=((1,0),(0,1)),phases=(0.,0.),species=0,detector=(1,1),detector_phase=0.):
    return dict(means=list(means),amplitudes=list(amplitudes),waves=[list(k) for k in waves],phases=list(phases),species=species,detector=list(detector),detector_phase=detector_phase)


def calibration_inputs():
    return [experiment(amplitudes=(a,b),waves=((1,0),(2,0)),species=s,detector=(q,0)) for a,b in [(.3,.2),(.6,-.4),(-.65,.65)] for s in [0,1,2] for q in [1,2,3]]


def hidden_inputs():
    return {'crossed_gradients':[experiment(species=s,detector=q) for s in [0,1,2] for q in [(1,1),(1,-1),(2,1),(1,2)]],
            'oblique_gradients':[experiment(means=(.8,1.2),amplitudes=(.58,-.8),waves=((1,1),(-1,1)),phases=(.3,-.4),species=s,detector=q,detector_phase=.2) for s in [0,1,2] for q in [(2,0),(0,2),(3,1),(1,3)]],
            'phase_shifted':[experiment(means=(1.1,.9),amplitudes=(.8,.65),waves=((1,0),(0,1)),phases=(.4,-.7),species=s,detector=q,detector_phase=-.3) for s in [0,1,2] for q in [(1,1),(1,-1),(2,1),(1,2)]]}


@lru_cache(128)
def fields(key,points=96):
    e=json.loads(key);x,y=np.meshgrid(np.arange(points)*2*np.pi/points,np.arange(points)*2*np.pi/points,indexing='ij');h=2*np.pi/points
    c=[]
    for i in [0,1]:
        k=e['waves'][i];c.append(e['means'][i]+e['amplitudes'][i]*np.cos(k[0]*x+k[1]*y+e['phases'][i]))
    c=np.array([c[0],c[1],c[0]+c[1]]);d=np.array([1.,.08,3.]);z=np.array([1.,1.,-1.])
    conductivity=np.einsum('i,ixy->xy',d,c);g=np.einsum('i,i,ixy->xy',d,z,c)
    ix=np.arange(points**2).reshape(points,points);rows=[];cols=[];values=[];rhs=np.zeros_like(g)
    for axis in [0,1]:
        face=(conductivity+np.roll(conductivity,-1,axis=axis))/2/h**2
        neighbor=np.roll(ix,-1,axis=axis)
        for a,b,v in [(ix,ix,face),(ix,neighbor,-face),(neighbor,ix,-face),(neighbor,neighbor,face)]:
            rows.extend(a.ravel());cols.extend(b.ravel());values.extend(v.ravel())
        rhs+=(np.roll(g,-1,axis=axis)+np.roll(g,1,axis=axis)-2*g)/h**2
    matrix=coo_matrix((values,(rows,cols)),shape=(points**2,points**2)).tocsc()
    # Fix one potential value. The omitted equation follows from conservation.
    potential=np.zeros(points**2);potential[1:]=spsolve(matrix[1:,1:],rhs.ravel()[1:]);potential=potential.reshape(points,points)
    change=np.zeros_like(c);current=[];electric=[]
    for axis in [0,1]:
        field=-(np.roll(potential,-1,axis=axis)-potential)/h;electric.append(field)
        flux=-d[:,None,None]*(np.roll(c,-1,axis=axis+1)-c)/h+d[:,None,None]*z[:,None,None]*(c+np.roll(c,-1,axis=axis+1))/2*field
        change-=(flux-np.roll(flux,1,axis=axis+1))/h
        current.append(np.einsum('i,ixy->xy',z,flux))
    return x,y,change,np.array(current),np.array(electric)


def predict(experiments,diffusivity,points=96):
    result=[]
    for e in experiments:
        if e['detector'][1]==0 and all(k[1]==0 for k in e['waves']) and all(p==0 for p in e['phases']):
            # Exact one-dimensional reflection-symmetric calibration reduction.
            n=512;x=np.arange(n)*2*np.pi/n;c=[];gradient=[]
            for i in [0,1]:
                k=e['waves'][i][0];c.append(e['means'][i]+e['amplitudes'][i]*np.cos(k*x));gradient.append(-k*e['amplitudes'][i]*np.sin(k*x))
            c=np.array([c[0],c[1],c[0]+c[1]]);gradient=np.array([gradient[0],gradient[1],gradient[0]+gradient[1]])
            d=np.array([1.,.08,3.]);z=np.array([1.,1.,-1.]);potential_gradient=-np.einsum('i,i,ix->x',d,z,gradient)/np.einsum('i,ix->x',d,c)
            s=e['species'];flux=-d[s]*(gradient[s]+z[s]*c[s]*potential_gradient)
            q=e['detector'][0];value=2*np.mean(-q*np.sin(q*x+e['detector_phase'])*flux)
        else:
            prep={k:e[k] for k in ['means','amplitudes','waves','phases']};x,y,change,_,_=fields(json.dumps(prep,sort_keys=True),points)
            q=e['detector'];value=2*np.mean(change[e['species']]*np.cos(q[0]*x+q[1]*y+e['detector_phase']))
        result.append(diffusivity*value)
    return np.array(result)
