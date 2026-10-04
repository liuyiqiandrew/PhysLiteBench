from functools import lru_cache
import json
import numpy as np
from scipy.sparse.linalg import LinearOperator, cg

GRADIENT=.2


@lru_cache(None)
def grid(points):
    x=np.arange(points)*2*np.pi/points
    xx,yy=np.meshgrid(x,x,indexing='ij')
    k=np.fft.fftfreq(points,1/points)
    kx,ky=np.meshgrid(k,k,indexing='ij')
    square=kx*kx+ky*ky
    inverse=np.zeros_like(square);inverse[square>0]=1/square[square>0]
    return xx,yy,kx,ky,inverse


def gradient(value,kx,ky):
    z=np.fft.fft2(value)
    return np.array([np.fft.ifft2(1j*k*z).real for k in [kx,ky]])


def divergence(value,kx,ky):
    return np.fft.ifft2(1j*(kx*np.fft.fft2(value[0])+ky*np.fft.fft2(value[1]))).real


def project(value,kx,ky,inverse):
    z=np.fft.fft2(value,axes=(-2,-1))
    longitudinal=(kx*z[0]+ky*z[1])*inverse
    z-=np.array([kx,ky])*longitudinal
    z[:,0,0]=0.
    return np.fft.ifft2(z,axes=(-2,-1)).real


@lru_cache(128)
def unit_velocity(key,points=49):
    e=json.loads(key);x,y,kx,ky,inverse=grid(points)
    c=sum(m['amplitude']*np.cos(m['wave'][0]*x+m['wave'][1]*y+m['phase']) for m in e['modes'])
    viscosity=np.exp(e['viscosity_contrast']*c)
    lap=np.fft.ifft2(-(kx*kx+ky*ky)*np.fft.fft2(c)).real
    force=project(-GRADIENT*lap*gradient(c,kx,ky),kx,ky,inverse)
    def action(flat):
        u=flat.reshape(2,points,points)
        derivatives=np.array([gradient(u[i],kx,ky) for i in range(2)])
        stress=viscosity*(derivatives+derivatives.swapaxes(0,1))
        value=np.array([divergence(stress[i],kx,ky) for i in range(2)])
        return -project(value,kx,ky,inverse).ravel()
    def precondition(flat):
        z=np.fft.fft2(flat.reshape(2,points,points),axes=(-2,-1))
        return np.fft.ifft2(z*inverse/viscosity.mean(),axes=(-2,-1)).real.ravel()
    operator=LinearOperator((force.size,force.size),matvec=action,dtype=float)
    pre=LinearOperator(operator.shape,matvec=precondition,dtype=float)
    velocity,info=cg(operator,force.ravel(),M=pre,rtol=1e-11,atol=1e-13,maxiter=500)
    if info:raise RuntimeError('Stokes solve did not converge')
    return x,y,velocity.reshape(2,points,points)


def predict_at(experiments,viscosity):
    result=[]
    for e in experiments:
        prep={k:e[k] for k in ['modes','viscosity_contrast']}
        x,y,u=unit_velocity(json.dumps(prep,sort_keys=True))
        q=e['detector_wave'];weight=np.cos(q[0]*x+q[1]*y+e['detector_phase'])
        result.append(2*np.mean(u[e['component']]*weight)/viscosity)
    return np.array(result)


class Model:
    def __init__(self):self.viscosity=None

    def fit(self,records):
        unit=predict_at([r['input'] for r in records],1.)
        values=np.array([r['value'] for r in records]);sigma=np.array([r['sigma'] for r in records])
        inverse=np.sum(unit*values/sigma**2)/np.sum(unit**2/sigma**2)
        self.viscosity=float(np.clip(1/inverse,.002,.008))
        return self

    def predict(self,experiments):return predict_at(experiments,self.viscosity)
