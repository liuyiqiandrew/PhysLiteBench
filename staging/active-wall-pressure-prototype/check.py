"""Prototype only: stationary active-particle wall force versus bulk stress closure."""
from pathlib import Path
import json
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve


def potential(x,theta,L,k,h,ell):
    s=np.maximum(abs(x)-L,0)
    a=s*s/(ell*ell+s*s)
    ap=2*s*ell*ell/(ell*ell+s*s)**2
    V=.5*k*s*s+h*a*np.sin(theta)**2
    Vx=np.sign(x)*(k*s+h*ap*np.sin(theta)**2)
    Vtheta=h*a*np.sin(2*theta)
    return V,Vx,Vtheta


def bernoulli(z):
    out=np.empty_like(z)
    small=abs(z)<1e-6
    out[small]=1-z[small]/2+z[small]**2/12
    out[~small]=z[~small]/np.expm1(z[~small])
    return out


def solve(Dr=.8,v=1.5,T=.5,L=1.5,k=4,h=0,ell=.6,nx=161,nt=48,tail=5):
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
    rho=prob.sum(axis=1)/dx
    m=(prob*np.cos(tt)).sum(axis=1)/dx
    c2=(prob*np.cos(tt)**2).sum(axis=1)/dx
    mid=nx//2
    stress=T*rho[mid]+v/Dr*(v*c2[mid]-T*(m[mid+1]-m[mid-1])/(2*dx))
    force=float((Vx[mid+1:]*prob[mid+1:]).sum())
    torque=float((-Dr/T*Vt[mid+1:]*np.sin(tt)*prob[mid+1:]).sum())
    return {'force':force,'shortcut':float(stress),'moment_balance':float(stress-v/Dr*torque),'relative_gap':float((stress-force)/force),'minimum_cell_probability':float(prob.min()),'edge_mass':float(prob[[0,-1]].sum()),'normalization':float(prob.sum()),'stationarity_residual':float(np.max(abs(A@prob.ravel()))),'center_density':float(rho[mid])}


def main():
    report={'scope':'Unevaluated numerical prototype; no task package or model trials.','cases':[]}
    for h in [0,.5,1,2,-.5,-1]:
        case={'Dr':.8,'v':1.5,'T':.5,'L':1.5,'k':4,'h':h,'ell':.6}
        low=solve(**case,nx=161,nt=48)
        high=solve(**case,nx=321,nt=96)
        report['cases'].append({'inputs':case,'coarse':low,'fine':high})
    out=Path(__file__).with_name('report.json');out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
