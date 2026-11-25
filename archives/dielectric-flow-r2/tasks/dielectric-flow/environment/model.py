from functools import lru_cache
import json
import numpy as np
from scipy.sparse.linalg import LinearOperator, cg


@lru_cache(None)
def geometry(points):
    x=np.arange(points)*2*np.pi/points
    xx,yy=np.meshgrid(x,x,indexing='ij')
    modes=np.fft.fftfreq(points,1/points)
    kx,ky=np.meshgrid(modes,modes,indexing='ij')
    square=kx*kx+ky*ky
    inverse=np.zeros_like(square);inverse[square>0]=1/square[square>0]
    return xx,yy,kx,ky,inverse


def gradient(value,kx,ky):
    z=np.fft.fft2(value)
    return np.array([np.fft.ifft2(1j*kx*z).real,np.fft.ifft2(1j*ky*z).real])


def divergence(value,kx,ky):
    return np.fft.ifft2(1j*kx*np.fft.fft2(value[0])+1j*ky*np.fft.fft2(value[1])).real


def material(modes,x,y):
    return 1+sum(m['amplitude']*np.cos(m['wave'][0]*x+m['wave'][1]*y+m['phase']) for m in modes)


@lru_cache(128)
def electric_state(key,points=49):
    e=json.loads(key);x,y,kx,ky,inverse=geometry(points)
    epsilon=material(e['permittivity_modes'],x,y)+np.zeros_like(x)
    conductivity=100*(material(e['conductivity_modes'],x,y)+np.zeros_like(x))
    mean=np.array(e['field'])[:,None,None]
    def action(v):
        v=v.reshape(points,points)
        return (-divergence(conductivity*gradient(v,kx,ky),kx,ky)+v.mean()).ravel()
    def precondition(v):
        z=np.fft.fft2(v.reshape(points,points));z[0,0]=0
        return np.fft.ifft2(z*inverse/conductivity.mean()).real.ravel()
    operator=LinearOperator((points**2,points**2),matvec=action,dtype=float)
    pre=LinearOperator(operator.shape,matvec=precondition,dtype=float)
    rhs=-divergence(conductivity*mean,kx,ky)
    potential,info=cg(operator,rhs.ravel(),M=pre,rtol=1e-12,atol=1e-13,maxiter=400)
    if info:raise RuntimeError('Electrical potential did not converge')
    electric=mean-gradient(potential.reshape(points,points),kx,ky)
    return x,y,epsilon,conductivity,electric


def electric_force(epsilon,electric,kx,ky):
    force=-.5*np.sum(electric**2,axis=0)*gradient(epsilon,kx,ky)
    return force


@lru_cache(128)
def unit_velocity(key,points=49):
    x,y,epsilon,_,electric=electric_state(key,points)
    _,_,kx,ky,inverse=geometry(points)
    force=electric_force(epsilon,electric,kx,ky)
    z=np.fft.fft2(force,axes=(-2,-1))
    longitudinal=(kx*z[0]+ky*z[1])*inverse
    velocity=np.fft.ifft2((z-np.array([kx,ky])*longitudinal)*inverse,axes=(-2,-1)).real
    return x,y,velocity


def predict_at(experiments,viscosity):
    out=[]
    for e in experiments:
        prep={k:e[k] for k in ['permittivity_modes','conductivity_modes','field']}
        x,y,velocity=unit_velocity(json.dumps(prep,sort_keys=True))
        q=e['detector_wave'];weight=np.cos(q[0]*x+q[1]*y+e['detector_phase'])
        out.append(2*np.mean(velocity[e['component']]*weight)/viscosity)
    return np.array(out)


class Model:
    def __init__(self):self.viscosity=None

    def fit(self,records):
        raise NotImplementedError

    def predict(self,experiments):return predict_at(experiments,self.viscosity)
