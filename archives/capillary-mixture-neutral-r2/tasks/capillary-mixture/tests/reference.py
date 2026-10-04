"""Minimize Newtonian dissipation over a real divergence-free streamfunction basis."""
from functools import lru_cache
import json
import numpy as np
from scipy.linalg import cho_factor, cho_solve

TRUE_PARAMETER=.004


def mode(wave,amplitude,phase=0.):
    return {'wave':list(wave),'amplitude':float(amplitude),'phase':float(phase)}


def experiment(modes,contrast=0.,component=0,wave=(0,2),phase=-np.pi/2):
    return {'modes':modes,'viscosity_contrast':float(contrast),'component':int(component),
            'detector_wave':list(wave),'detector_phase':float(phase)}


def patterns():
    return [[mode((1,0),.18),mode((1,2),.14,.5)],
            [mode((1,1),.2,.3),mode((2,-1),.13,-.7)],
            [mode((2,0),.19,-.2),mode((1,1),.14,.9)]]


def projections(modes):
    k=np.array(modes[0]['wave']);l=np.array(modes[1]['wave'])
    return [(0 if abs(q[1])>=abs(q[0]) else 1,q.tolist()) for q in [k-l,k+l,k,l]]


def calibration_inputs():
    return [experiment(modes,0.,component,q,phase) for modes in patterns()
            for component,q in projections(modes)[:2] for phase in [0.,-np.pi/2]]*12


def hidden_inputs():
    return {name:[experiment(modes,beta,component,q,phase)
                  for beta in [3.5,5.5,7.] for component,q in projections(modes)
                  for phase in [0.,-np.pi/2]]
            for name,modes in zip(['crossed_waves','oblique_waves','unequal_waves'],patterns())}


@lru_cache(4)
def basis(cutoff,points):
    x=np.arange(points)*2*np.pi/points
    x,y=np.meshgrid(x,x,indexing='ij');x=x.ravel();y=y.ravel()
    wave=np.array([(a,b) for a in range(cutoff+1) for b in range(-cutoff,cutoff+1)
                   if a>0 or b>0])
    angle=x[:,None]*wave[:,0]+y[:,None]*wave[:,1]
    sine=np.sin(angle);cosine=np.cos(angle)
    psi=np.concatenate([sine,cosine],axis=1)
    derivative=np.concatenate([cosine,-sine],axis=1)
    k=np.tile(wave,(2,1))
    velocity_x=derivative*k[:,1]
    velocity_y=-derivative*k[:,0]
    strain_xx=-psi*k[:,0]*k[:,1]
    strain_xy=.5*psi*(k[:,0]**2-k[:,1]**2)
    return x,y,velocity_x,velocity_y,strain_xx,strain_xy


@lru_cache(64)
def amplitudes(key,cutoff=12,points=65):
    e=json.loads(key);x,y,ux,uy,dxx,dxy=basis(cutoff,points)
    c=np.zeros_like(x);cx=np.zeros_like(x);cy=np.zeros_like(x);mu=np.zeros_like(x)
    for m in e['modes']:
        a,b=m['wave'];angle=a*x+b*y+m['phase'];value=m['amplitude']*np.cos(angle)
        c+=value;mu+=(1+.2*(a*a+b*b))*value
        cx-=a*m['amplitude']*np.sin(angle);cy-=b*m['amplitude']*np.sin(angle)
    eta=np.exp(e['viscosity_contrast']*c)
    # For u=(psi_y,-psi_x), Dyy=-Dxx and 2 eta D:D
    # gives this positive quadratic form. The load uses mu*grad(c),
    # pressure-equivalent to the free-energy virtual-work force.
    stiffness=4*(dxx.T@(eta[:,None]*dxx)+dxy.T@(eta[:,None]*dxy))/len(x)
    force=(ux.T@(mu*cx)+uy.T@(mu*cy))/len(x)
    return cho_solve(cho_factor(stiffness,lower=True,check_finite=False),force,check_finite=False)


def predict(experiments,viscosity,cutoff=12,points=65):
    x,y,ux,uy,*_=basis(cutoff,points)
    out=[]
    for e in experiments:
        prep={k:e[k] for k in ['modes','viscosity_contrast']}
        coefficients=amplitudes(json.dumps(prep,sort_keys=True),cutoff,points)
        q=e['detector_wave'];detector=np.cos(q[0]*x+q[1]*y+e['detector_phase'])
        projection=2*(detector@(ux if e['component']==0 else uy))/len(x)
        out.append(float(projection@coefficients/viscosity))
    return np.array(out)
