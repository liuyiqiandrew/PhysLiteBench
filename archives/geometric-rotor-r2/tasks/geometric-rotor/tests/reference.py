"""Independent finite-gap laboratory Fourier Hamiltonian and kinetic trace."""
from functools import lru_cache
import numpy as np
from scipy.linalg import eigh,expm

TRUE_PARAMETER=1.07
X=np.array([[0,1],[1,0]],complex)
Z=np.diag([1.,-1.])
Q=np.diag([0.,0.,1.,1.])


@lru_cache(maxsize=256)
def potential(theta,q,r,cutoff):
    first=q*np.kron(np.eye(2),X)/2
    second=r*np.kron(np.cos(theta)*Z+np.sin(theta)*X,Z)/2
    samples=[]
    for phi in 2*np.pi*np.arange(128)/128:
        u=expm(-1j*phi*first)@expm(-1j*phi*second)
        samples.append(u@Q@u.conj().T)
    coefficients=np.fft.fft(samples,axis=0)/128
    n=np.arange(-cutoff,cutoff+1)
    matrix=np.empty((4*len(n),4*len(n)),complex)
    for i,a in enumerate(n):
        for j,b in enumerate(n):matrix[4*i:4*i+4,4*j:4*j+4]=coefficients[(a-b)%128]
    return n,matrix


@lru_cache(maxsize=1024)
def modes(theta,q,r,inertia,gap,cutoff):
    n,v=potential(theta,q,r,cutoff)
    kinetic=np.repeat(n*n/(2*inertia),4)
    energy,vectors=eigh(np.diag(kinetic)+gap*v)
    expectation=abs(vectors).T**2@kinetic
    return energy,expectation


def finite(e,inertia,gap,cutoff=18):
    energy,kinetic=modes(e['theta'],e['q'],e['r'],inertia,gap,cutoff)
    weight=np.exp(-(energy-energy.min())/e['temperature'])
    return float(weight@kinetic/weight.sum())


def response(e,inertia,base_gap=1024,cutoff=18):
    values=[finite(e,inertia,base_gap*f,cutoff) for f in [1,2,4]]
    return (values[0]-6*values[1]+8*values[2])/3


def predict(experiments,inertia=TRUE_PARAMETER):
    return np.array([response(e,inertia) for e in experiments])


def experiment(theta=.65,temperature=.08,q=2,r=1):
    return dict(theta=theta,temperature=temperature,q=q,r=r)


def calibration_inputs():
    return [experiment(theta=np.pi/2,temperature=t,q=q,r=r) for _ in range(16)
            for q in [1,2,3] for r in [1,2] for t in [.06,.15,.28]]


def hidden_inputs():
    return {'tilt_sweep':[experiment(theta=t,temperature=.07) for t in [.45,.6,.75,.9]],
            'temperature_sweep':[experiment(temperature=t) for t in [.05,.08,.11,.14]],
            'textures':[experiment(theta=.45,temperature=.1,q=1,r=2),experiment(theta=.4,temperature=.1,q=3,r=2),experiment(theta=.8,temperature=.05,q=2,r=1),experiment(theta=.75,temperature=.07,q=1,r=2)],
            'central_holonomy_anchors':[experiment(theta=np.pi/2,temperature=.1,q=1,r=1),experiment(theta=np.pi/2,temperature=.2,q=2,r=1)]}
