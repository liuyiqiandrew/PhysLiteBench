"""Minimize the six-component strain energy subject to compatibility."""
from functools import lru_cache
import numpy as np

PARAMETER='mobility'
TRUE_PARAMETER=.085


def experiment(q=(0,1,1),time=1.,amplitude=.02):
    return dict(wavevector=list(q),time=float(time),amplitude=float(amplitude))


def calibration_inputs():
    return [experiment((k,0,0),t,a) for k in [1,2,3] for a in [.015,-.025] for t in np.linspace(.05,5.,20)]


def hidden_inputs():
    return {'compatible_shear':[experiment(q,t) for q in [(0,1,1),(0,1,-1),(0,-1,1)] for t in np.linspace(.1,6.,24)],
            'oblique_shear':[experiment(q,t,-.025) for q in [(0,2,1),(0,1,2),(0,-2,1)] for t in np.linspace(.1,4.,24)],
            'three_dimensional':[experiment(q,t) for q in [(1,1,1),(-1,1,1),(1,1,-1)] for t in np.linspace(.1,5.,24)]}


@lru_cache(64)
def minimum_strain(q):
    n=np.asarray(q,dtype=float);n/=np.linalg.norm(n)
    axis=np.array([1.,0.,0.]) if abs(n[0])<.9 else np.array([0.,1.,0.])
    t=np.cross(n,axis);t/=np.linalg.norm(t);s=np.cross(n,t)
    tensors=[]
    for i,j in [(0,0),(1,1),(2,2),(1,2),(0,2),(0,1)]:
        e=np.zeros((3,3));e[i,j]=1;e[j,i]=1;tensors.append(e)
    constraints=np.array([[t@e@t for e in tensors],[s@e@s for e in tensors],[t@e@s for e in tensors]])
    # Hessian in tensor strain components xx,yy,zz,yz,xz,xy.
    h=np.diag([6.,6.,6.,12.,12.,12.]);h[:3,:3]+=2.
    e0=np.array([0.,.5,-.5,0.,0.,0.])
    block=np.block([[h,constraints.T],[constraints,np.zeros((3,3))]])
    sol=np.linalg.solve(block,np.r_[h@e0,np.zeros(3)])
    strain=sol[:6];difference=strain-e0
    # Twice the minimized elastic energy for unit concentration.
    penalty=float(difference@h@difference)
    stress=h@difference
    return penalty,strain,constraints,stress


def predict(experiments,mobility):
    out=[]
    for e in experiments:
        square=float(np.dot(e['wavevector'],e['wavevector']))
        penalty=minimum_strain(tuple(e['wavevector']))[0]
        rate=mobility*square*(.4+.2*square+penalty)
        out.append(e['amplitude']*np.exp(-rate*e['time']))
    return np.array(out)
