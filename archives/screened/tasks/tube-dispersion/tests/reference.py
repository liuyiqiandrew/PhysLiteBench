"""Annular finite-volume axial moment equations for reflecting transverse diffusion."""
from functools import lru_cache
import numpy as np
from scipy.linalg import eigh

PARAMETER = 'diffusivity'
TRUE_PARAMETER = .02


def experiment(t,radius=.5,flow=1.,width=.1,observable='axial_variance'):
    return dict(time=float(t),radius=float(radius),mean_flow=float(flow),initial_width=float(width),observable=observable)


def calibration_inputs():
    return [experiment(t,radius=a,flow=0.,width=w) for a,w in [(.3,.1),(.5,.15),(.6,.08),(.4,.12)]
            for t in np.linspace(.05,4.,25)]


def hidden_inputs():
    return {'startup_dispersion':[experiment(t) for t in np.geomspace(.03,2.,20)],
            'small_tube':[experiment(t,radius=.25,flow=.7) for t in np.geomspace(.02,3.,20)],
            'reversed_flow':[experiment(t,radius=.6,flow=-.9,width=.08) for t in np.geomspace(.05,5.,20)]}


@lru_cache(8)
def radial(cells):
    edge=np.linspace(0.,1.,cells+1); dx=1/cells
    weight=np.diff(edge**2)
    flux=edge[1:-1]/dx
    matrix=np.zeros((cells,cells))
    for i,f in enumerate(flux):
        matrix[i,i]-=2*f/weight[i];matrix[i,i+1]+=2*f/weight[i]
        matrix[i+1,i]+=2*f/weight[i+1];matrix[i+1,i+1]-=2*f/weight[i+1]
    speed=2-edge[:-1]**2-edge[1:]**2
    symmetric=np.sqrt(weight)[:,None]*matrix/np.sqrt(weight)[None,:]
    values,vectors=eigh(-symmetric)
    coefficients=vectors[:,1:].T@(np.sqrt(weight)*(speed-1))
    return weight,matrix,speed,values[1:],coefficients


def predict(experiments,diffusivity,cells=384):
    weight,matrix,speed,eigenvalues,coefficients=radial(cells)
    out=[]
    for e in experiments:
        t,U,a=e['time'],e['mean_flow'],e['radius']
        # The first moment solves m1dot=D*L*m1+u. Integrating the
        # area mean of m2dot=D*L*m2+2*u*m1+2D gives the expression below.
        rates=diffusivity*eigenvalues/a**2
        z=rates*t
        integral=np.where(z<1e-4,z*z*(.5-z/6+z*z/24),z+np.expm1(-z))/rates**2
        variance=e['initial_width']**2+2*diffusivity*t+2*U*U*np.sum(coefficients**2*integral)
        out.append(U*t if e['observable']=='mean_position' else variance)
    return np.array(out)
