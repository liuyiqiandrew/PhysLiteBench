import numpy as np
from scipy.integrate import solve_bvp


def exact(xi,f,z):
    d=1-f+f/z
    if xi<=1/z:
        free=xi*d; residual=xi
    elif xi*(1-f)<=1/z:
        residual=1/z; free=xi*(1-f)*(1-1/z)+1/z**2
    elif xi*(1-f)<=1:
        residual=xi*(1-f); free=residual
    else:
        residual=1.;free=1.
    return (free-residual**2/2)/(xi*d)


def finite(xi,f,z,length,previous=None,old_L=None):
    # w_i=log(4*pi*lB*r^2*n_i); E=d psi/d log r, psi=e Phi/kBT.
    s=np.unique(np.r_[np.linspace(0,1,401),np.geomspace(1e-5,.1,100),1-np.geomspace(1e-5,.1,100)])
    if previous is None:
        w1=np.full_like(s,np.log(2*xi*(1-f)/length))
        wh=np.full_like(s,np.log(2*xi*f/(z*length)))
        vals=np.array([w1,wh,2*xi*(s-1),2*xi*f*s])
    else:
        vals=previous.sol(s)
    def rhs(s,y):
        v1=np.exp(y[0]); vh=np.exp(y[1])
        return length*np.vstack([2+y[2],2+z*y[2],v1+z*vh,z*vh])
    def bc(left,right):
        return [left[2]+2*xi,right[2],left[3],right[3]-2*xi*f]
    sol=solve_bvp(rhs,bc,s,vals,tol=1e-7,max_nodes=30000)
    assert sol.success,sol.message
    phi=(np.exp(sol.y[0,-1])+np.exp(sol.y[1,-1]))*(-np.expm1(-2*length))/(4*xi*(1-f+f/z))
    return phi,sol


def solve(xi,f,z,length):
    sol=None
    for L in np.geomspace(.1,length,100):
        p,sol=finite(xi,f,z,L,sol)
    return p


if __name__=='__main__':
    for xi,f,z in [(.2,.5,2),(.7,.5,2),(1.2,.5,2),(2.5,.5,2),(.5,.5,3),(1.,.3,3),(1.,.8,3),(1.6,.5,3)]:
        for L in [20,80,320]:
            p=solve(xi,f,z,L)
            print(xi,f,z,L,p,'exact',exact(xi,f,z),'error',p-exact(xi,f,z),flush=True)
