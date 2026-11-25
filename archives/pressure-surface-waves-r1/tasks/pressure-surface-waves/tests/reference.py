"""Surface eigenmode from scalar-energy second variations and depth finite elements."""
from functools import lru_cache
import itertools
import numpy as np
from scipy.optimize import brentq
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import eigsh

TRUE_PARAMETER=1100.
MU=1.;LAM=2.

def stretch(s,p):
    return brentq(lambda t:MU*(t*t-1)+LAM*np.log(s*t)+p*s*t,.2,2.)


def energy(F,p):
    j=np.linalg.det(F)
    return MU/2*(np.sum(F*F)-3-2*np.log(j))+LAM/2*np.log(j)**2+p*j


def hessian(s,p,follower):
    t=stretch(s,p);F=np.diag([s,1.,t]);J=np.linalg.det(F)
    maps=[]
    for i in [0,2]:
        for a in [0,2]:
            L=np.zeros((3,3));L[i,a]=1
            maps.append(L@F)
    def calc(h):
        A=np.empty((4,4))
        for i,ei in enumerate(maps):
            for j,ej in enumerate(maps):
                A[i,j]=(energy(F+h*(ei+ej),p if follower else 0)-energy(F+h*(ei-ej),p if follower else 0)-energy(F+h*(-ei+ej),p if follower else 0)+energy(F-h*(ei+ej),p if follower else 0))/(4*h*h*J)
        return A
    return (4*calc(.0002)-calc(.0004))/3


def finite_element(s,p,rho=1.,follower=True,n=384,depth=28.):
    h=depth/n;A=hessian(s,p,follower);J=s*stretch(s,p)
    size=2*(n+1);K=lil_matrix((size,size),dtype=complex);M=lil_matrix((size,size),dtype=complex)
    for element in range(n):
        inds=[2*element,2*element+1,2*element+2,2*element+3]
        ke=np.zeros((4,4),complex);me=np.zeros((4,4),complex)
        for z in [-1/np.sqrt(3),1/np.sqrt(3)]:
            N=np.array([(1-z)/2,(1+z)/2]);dN=np.array([-1,1])/h
            B=np.zeros((4,4),complex);Nmat=np.zeros((2,4),complex)
            for node in range(2):
                for i in range(2):
                    B[2*i,2*node+i]=1j*N[node];B[2*i+1,2*node+i]=dN[node]
                    Nmat[i,2*node+i]=N[node]
            ke+=h/2*B.conj().T@A@B
            me+=h/2*rho/J*Nmat.conj().T@Nmat
        for i in range(4):
            for j in range(4):K[inds[i],inds[j]]+=ke[i,j];M[inds[i],inds[j]]+=me[i,j]
    K=K[:-2,:-2].tocsc();M=M[:-2,:-2].tocsc()
    v2=eigsh(K,k=1,M=M,sigma=0.,which='LM',tol=1e-11,return_eigenvectors=False,v0=np.ones(K.shape[0]))
    return np.sqrt(min(v2.real))



@lru_cache(128)
def unit_speed(stretch, pressure, cells=384, depth=28.):
    coarse=finite_element(stretch,pressure,follower=True,n=cells,depth=depth)
    fine=finite_element(stretch,pressure,follower=True,n=2*cells,depth=depth)
    return (4*fine-coarse)/3


def predict(experiments,density=TRUE_PARAMETER,cells=384,depth=28.):
    return np.array([unit_speed(e['stretch'],e['pressure']/1e6,cells,depth)*np.sqrt(1e6/density) for e in experiments])


def experiment(stretch=1.,pressure=700000.):
    return dict(stretch=stretch,pressure=pressure)


def calibration_inputs():
    return [experiment(s,0.) for s,repeat in itertools.product([.95,1.,1.05,1.1,1.15,1.2],range(40))]


def hidden_inputs():
    return {
        'pressure_sweep':[experiment(1.,p) for p in [500000.,600000.,700000.,800000.]],
        'stretch_sweep':[experiment(s,750000.) for s in [.95,1.,1.1,1.2]],
        'loaded_waves':[experiment(s,p) for s,p in [(1.02,650000.),(1.08,700000.),(.98,780000.),(1.16,800000.)]],
        'unloaded_anchors':[experiment(s,0.) for s in [.975,1.075,1.125,1.175]]
    }
