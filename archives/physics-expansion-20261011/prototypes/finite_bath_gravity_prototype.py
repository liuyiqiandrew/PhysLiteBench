import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from scipy.special import roots_legendre
from scipy.linalg import eigh


def emden():
    eps=1e-5
    def rhs(x,y):
        return [y[1],np.exp(-y[0])-2*y[1]/x]
    return solve_ivp(rhs,(eps,60.),[eps**2/6,eps/3],rtol=2e-12,atol=2e-14,dense_output=True,max_step=.08).sol


PROFILE=emden()


def milne(alpha):
    psi,gradient=PROFILE(alpha)
    v=alpha*gradient
    u=alpha**2*np.exp(-psi)/v
    return u,v


def turning(bath):
    if bath is None:
        return brentq(lambda alpha:milne(alpha)[0]-1,5.,12.)
    def equation(alpha):
        u,v=milne(alpha)
        return 4*u*u+2*u*v+(2*bath-11)*u+3-2*bath
    return brentq(equation,8.99,40.)


def hessian(alpha,bath,cells=128):
    # Piecewise radial mass perturbations q(0)=q(1)=0. First element uses
    # q~r^3 to keep the corresponding density perturbation finite at r=0.
    _,v=milne(alpha);theta=1/v
    z,w=roots_legendre(8)
    edge=np.linspace(0.,1.,cells+1)
    entropy=np.zeros((cells+1,cells+1))
    grav=np.zeros_like(entropy);energy=np.zeros(cells+1)
    for i in range(cells):
        left,right=edge[i:i+2];step=right-left
        r=left+(z+1)*step/2;weight=w*step/2
        psi,gradient=PROFILE(alpha*r)
        density=alpha**2*theta*np.exp(-psi)/(4*np.pi)
        mass=alpha*r*r*theta*gradient
        if i==0:
            shape=np.array([np.zeros_like(r),(r/right)**3])
            derivative=np.array([np.zeros_like(r),3*r*r/right**3])
        else:
            shape=np.array([(right-r)/step,(r-left)/step])
            derivative=np.array([-np.ones_like(r)/step,np.ones_like(r)/step])
        entropy[i:i+2,i:i+2]+=(derivative*(weight/(4*np.pi*density*r*r)))@derivative.T
        grav[i:i+2,i:i+2]+=(shape*(weight/(r*r)))@shape.T
        energy[i:i+2]+=shape@(weight*mass/(r*r))
    entropy=entropy[1:-1,1:-1];grav=grav[1:-1,1:-1];energy=energy[1:-1]
    curvature=entropy-grav/theta
    if bath is not None:
        curvature+=2*np.outer(energy,energy)/((3+2*bath)*theta**2)
    return float(eigh(curvature,entropy,subset_by_index=[0,0],check_finite=False)[0][0])


def reference_critical(bath,cells=128):
    alpha=brentq(lambda a:hessian(a,bath,cells),6.,45.,xtol=1e-8)
    return 1/milne(alpha)[1],alpha


if __name__=='__main__':
    for b in [None,0.,.3,.7,1.,1.5,2.]:
        a=turning(b);v=milne(a)[1];contrast=np.exp(PROFILE(a)[0]);t=1/v
        r128=reference_critical(b,128)
        r256=reference_critical(b,256)
        extrap=(4*r256[0]-r128[0])/3
        print(b,'alpha',a,'contrast',contrast,'temperature',t,'hessian',hessian(a,b,128),'refs',r128,r256,'extrap error',extrap-t,flush=True)
