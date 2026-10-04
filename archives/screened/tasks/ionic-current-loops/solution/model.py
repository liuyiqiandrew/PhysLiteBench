from functools import lru_cache
import json
import numpy as np
from scipy.sparse.linalg import LinearOperator, cg

RATIOS=np.array([1.,.08,3.])
CHARGES=np.array([1.,1.,-1.])


def preparation(e,points):
    x,y=np.meshgrid(2*np.pi*np.arange(points)/points,2*np.pi*np.arange(points)/points,indexing='ij')
    c=[]
    for i in [0,1]:
        k=e['waves'][i];c.append(e['means'][i]+e['amplitudes'][i]*np.cos(k[0]*x+k[1]*y+e['phases'][i]))
    c=np.array([c[0],c[1],c[0]+c[1]])
    return x,y,c


@lru_cache(256)
def rates(key,points=49):
    e=json.loads(key);x,y,c=preparation(e,points)
    k= np.fft.fftfreq(points,1/points)
    kx,ky=np.meshgrid(k,k,indexing='ij')
    def grad(a):
        hat=np.fft.fft2(a)
        return np.array([np.fft.ifft2(1j*kx*hat).real,np.fft.ifft2(1j*ky*hat).real])
    def div(v):
        return np.fft.ifft2(1j*kx*np.fft.fft2(v[0])+1j*ky*np.fft.fft2(v[1])).real
    conductivity=np.einsum('i,ixy->xy',RATIOS,c)
    diffusion_charge=np.einsum('i,i,ixy->xy',CHARGES,RATIOS,c)
    wave_square=kx*kx+ky*ky
    def action(v):
        v=v.reshape(points,points)
        return (-div(conductivity*grad(v))+np.mean(v)).ravel()
    def precondition(v):
        hat=np.fft.fft2(v.reshape(points,points));denom=wave_square*np.mean(conductivity);denom[0,0]=1.
        return np.fft.ifft2(hat/denom).real.ravel()
    operator=LinearOperator((points**2,points**2),matvec=action,dtype=float)
    pre=LinearOperator(operator.shape,matvec=precondition,dtype=float)
    rhs=div(grad(diffusion_charge)).ravel()
    potential,info=cg(operator,rhs,M=pre,rtol=2e-12,atol=1e-13,maxiter=300)
    if info:raise RuntimeError('Potential solve failed')
    potential_gradient=grad(potential.reshape(points,points))
    flux=np.array([-RATIOS[i]*(grad(c[i])+CHARGES[i]*c[i]*potential_gradient) for i in range(3)])
    return x,y,np.array([-div(j) for j in flux])


def predict_at(experiments,diffusivity):
    out=[]
    for e in experiments:
        prep={k:e[k] for k in ['means','amplitudes','waves','phases']}
        x,y,change=rates(json.dumps(prep,sort_keys=True))
        q=e['detector'];weight=np.cos(q[0]*x+q[1]*y+e['detector_phase'])
        out.append(2*diffusivity*np.mean(change[e['species']]*weight))
    return np.array(out)


class Model:
    def __init__(self):self.diffusivity=None

    def fit(self,records):
        unit=predict_at([r['input'] for r in records],1.)
        value=np.array([r['value'] for r in records]);sigma=np.array([r['sigma'] for r in records])
        self.diffusivity=float(np.sum(unit*value/sigma**2)/np.sum(unit**2/sigma**2))
        return self

    def predict(self,experiments):return predict_at(experiments,self.diffusivity)
