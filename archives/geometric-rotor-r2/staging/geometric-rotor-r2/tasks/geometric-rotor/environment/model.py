from functools import lru_cache
import numpy as np
from scipy.linalg import eigh


I2=np.eye(2)
X=np.array([[0,1],[1,0]],complex)
Z=np.diag([1.,-1.])


def frame(phi,theta,q,r):
    g=np.kron(I2,X)
    j=np.kron(np.cos(theta)*Z+np.sin(theta)*X,Z)
    first=np.cos(q*phi/2)*np.eye(4)-1j*np.sin(q*phi/2)*g
    second=np.cos(r*phi/2)*np.eye(4)-1j*np.sin(r*phi/2)*j
    u=first@second
    derivative=(-1j*q/2*g)@u+first@(-1j*r/2*j)@second
    return u,derivative


@lru_cache(maxsize=512)
def levels(theta,q,r):
    samples=[]
    for phi in 2*np.pi*np.arange(128)/128:
        u,derivative=frame(phi,theta,q,r)
        low=u[:,:2]
        dlow=derivative[:,:2]
        samples.append(dlow.conj().T@(np.eye(4)-low@low.conj().T)@dlow)
    coefficients=np.fft.fft(samples,axis=0)/len(samples)
    n=np.arange(-18,19)
    shift=((q+r)%2)/2
    matrix=np.diag(np.repeat((n-shift)**2,2)).astype(complex)
    for i,ni in enumerate(n):
        for j,nj in enumerate(n):
            matrix[2*i:2*i+2,2*j:2*j+2]+=coefficients[(ni-nj)%len(samples)]
    return eigh(matrix/2,eigvals_only=True)


def predict_at(experiments,inertia):
    result=[]
    for e in experiments:
        energy=levels(e['theta'],e['q'],e['r'])/inertia
        weight=np.exp(-(energy-energy.min())/e['temperature'])
        result.append(float(weight@energy/weight.sum()))
    return np.array(result)


class Model:
    def __init__(self):
        self.inertia=None

    def fit(self,records):
        raise NotImplementedError

    def predict(self,experiments):
        return predict_at(experiments,self.inertia)
