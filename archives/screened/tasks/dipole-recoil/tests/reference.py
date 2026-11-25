"""Integrate angular radiated momentum directly in rationalized vacuum units."""
from functools import lru_cache
import numpy as np
from numpy.polynomial.legendre import leggauss

PARAMETER='polarizability_scale'
TRUE_PARAMETER=1.8


def experiment(k=1.,electric_weight=1.,magnetic_weight=1.):
    return dict(k=float(k),electric_weight=float(electric_weight),magnetic_weight=float(magnetic_weight))


def calibration_inputs():
    return [experiment(k,w,0.) if kind==0 else experiment(k,0.,w) for kind in [0,1] for w in [.6,1.,1.4] for k in np.linspace(.5,1.5,20)]


def hidden_inputs():
    return {'matched_dipoles':[experiment(k,w,w) for w in [.6,1.,1.4] for k in np.linspace(.6,1.6,12)],
            'electric_dominant':[experiment(k,1.4,w) for w in [.4,.7,1.] for k in np.linspace(.6,1.6,12)],
            'magnetic_dominant':[experiment(k,w,1.5) for w in [.4,.8,1.2] for k in np.linspace(.6,1.6,12)]}


@lru_cache(4)
def directions(order):
    z,w=leggauss(order);phi=np.arange(2*order)*np.pi/order
    zz,pp=np.meshgrid(z,phi,indexing='ij')
    n=np.column_stack([np.sqrt(1-zz.ravel()**2)*np.cos(pp.ravel()),np.sqrt(1-zz.ravel()**2)*np.sin(pp.ravel()),zz.ravel()])
    weights=np.repeat(w,2*order)*np.pi/order
    return n,weights


def radiation(e,scale,order=12):
    n,w=directions(order)
    alpha0=scale*np.array([e['electric_weight']]*3+[e['magnetic_weight']]*3)
    drive=np.array([1.,0.,0.,0.,1.,0.])
    moment=np.linalg.solve(np.eye(6)-1j*e['k']**3/(6*np.pi)*np.diag(alpha0),alpha0*drive)
    p,m=moment[:3],moment[3:]
    farfield=p[None,:]-n*(n@p)[:,None]+np.cross(m[None,:],n)
    intensity=e['k']**4/(32*np.pi**2)*np.sum(abs(farfield)**2,axis=1)
    return n,w,intensity


def predict(experiments,scale,order=12):
    out=[]
    for e in experiments:
        n,w,intensity=radiation(e,scale,order)
        # Losslessness equates removed incident energy to the total scattered
        # power. Momentum is instead weighted by the outgoing direction.
        out.append(float(np.sum(w*(1-n[:,2])*intensity)))
    return np.array(out)
