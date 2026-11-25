import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
from functools import lru_cache


def potential(x,theta,L,k,h,ell):
    s=np.maximum(abs(x)-L,0)
    a=s**4/(ell**4+s**4)
    ap=4*s**3*ell**4/(ell**4+s**4)**2
    V=.25*k*s**4+h*a*np.sin(theta)**2
    Vx=np.sign(x)*(k*s**3+h*ap*np.sin(theta)**2)
    Vtheta=h*a*np.sin(2*theta)
    return V,Vx,Vtheta


def bernoulli(z):
    out=np.empty_like(z)
    small=abs(z)<1e-6
    out[small]=1-z[small]/2+z[small]**2/12
    out[~small]=z[~small]/np.expm1(z[~small])
    return out


def solve(Dr=.8,v=1.5,T=.5,L=1.5,k=4,h=0,ell=.6,nx=161,nt=48,tail=3):
    X=L+tail
    x=np.linspace(-X,X,nx);dx=x[1]-x[0]
    theta=np.arange(nt)*2*np.pi/nt;dt=2*np.pi/nt
    xx=x[:,None];tt=theta[None,:]
    V,Vx,Vt=potential(xx,tt,L,k,h,ell)
    ids=np.arange(nx*nt).reshape(nx,nt)
    left=ids[:-1].ravel();right=ids[1:].ravel()
    delta=(v*np.cos(tt)*dx-np.diff(V,axis=0))/T
    fwd=T/dx**2*bernoulli(-delta).ravel();back=T/dx**2*bernoulli(delta).ravel()
    angular_left=ids.ravel();angular_right=np.roll(ids,-1,axis=1).ravel()
    delta_theta=-(np.roll(V,-1,axis=1)-V)/T
    afwd=Dr/dt**2*bernoulli(-delta_theta).ravel();aback=Dr/dt**2*bernoulli(delta_theta).ravel()
    a=np.r_[left,right,angular_left,angular_right]
    b=np.r_[right,left,angular_right,angular_left]
    rates=np.r_[fwd,back,afwd,aback]
    A=coo_matrix((np.r_[rates,-rates],(np.r_[b,a],np.r_[a,a])),shape=(nx*nt,)*2).tocsc()
    j=(nx//2)*nt
    # Fix one positive cell then normalize; all other stationarity equations remain.
    Q=A.tolil();Q[j,:]=0;Q[j,j]=1;rhs=np.zeros(nx*nt);rhs[j]=1
    prob=spsolve(Q.tocsc(),rhs);prob/=prob.sum();prob=prob.reshape(nx,nt)
    force=float((Vx[nx//2+1:]*prob[nx//2+1:]).sum())
    return force


@lru_cache(maxsize=256)
def unit_pressure(speed,temperature,half_width,alignment,refine=1):
    params=dict(v=speed,T=temperature,L=half_width,h=alignment)
    low=solve(**params,nx=160*refine+1,nt=48*refine)
    high=solve(**params,nx=320*refine+1,nt=96*refine)
    return (4*high-low)/3


def predict(experiments,loading):
    return np.asarray([loading*unit_pressure(*[float(e[key]) for key in
                       ('speed','temperature','half_width','alignment')]) for e in experiments])


TRUE_PARAMETER=1.07
SIGMA=.001


def calibration_inputs():
    unique=[dict(speed=v,temperature=t,half_width=L,alignment=0.)
            for v in [.9,1.4,1.8] for t in [.45,.7] for L in [1.1,1.9]]
    return unique*24


def hidden_inputs():
    def cases(rows):
        return [dict(speed=v,temperature=t,half_width=L,alignment=h) for v,t,L,h in rows]
    return {
        'normal_alignment':cases([(1.2,.5,1.2,1.0),(1.5,.6,1.6,1.5),(1.8,.5,1.9,2.)]),
        'tangential_alignment':cases([(1.2,.5,1.2,-.8),(1.5,.6,1.6,-1.),(1.8,.5,1.9,-1.)]),
        'mixed_controls':cases([(1.7,.45,1.,1.8),(1.1,.7,2.,-1.),(1.6,.65,1.4,1.2)]),
        'torque_free_anchors':cases([(1.1,.55,1.3,0.),(1.6,.65,1.7,0.)])}
