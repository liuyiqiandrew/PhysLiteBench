import numpy as np
from scipy.linalg import eigh
from functools import lru_cache

SITES=8
OCCUPIED={2:(3,4),3:(2,3,4),4:(2,3,4,5)}


def potential(amplitude,asymmetry):
    angle=2*np.pi*np.arange(SITES)/SITES
    return amplitude*np.cos(angle)+asymmetry*np.sin(angle)


def hamiltonian(hopping,flux,amplitude,asymmetry):
    matrix=np.diag(potential(amplitude,asymmetry)).astype(complex)
    for j in range(SITES):
        k=(j+1)%SITES
        bond=-hopping*np.exp(1j*flux/SITES)
        matrix[j,k]=bond
        matrix[k,j]=bond.conjugate()
    return matrix


@lru_cache(maxsize=16384)
def density(hopping,number,flux,amplitude,asymmetry,duration):
    energies,vectors=eigh(hamiltonian(hopping,flux,amplitude,asymmetry))
    propagator=(vectors*np.exp(-1j*energies*duration))@vectors.conj().T
    return np.sum(abs(propagator[:,OCCUPIED[number]])**2,axis=1)


def predict_at(experiments,hopping):
    return np.asarray([density(float(hopping),int(e['number']),float(e['flux']),float(e['amplitude']),float(e['asymmetry']),float(e['duration']))[e['site']] for e in experiments])


class Model:
    def __init__(self):
        self.hopping=None

    def fit(self,records):
        from scipy.optimize import minimize_scalar
        experiments=[r['input'] for r in records]
        values=np.array([r['value'] for r in records])
        sigma=np.array([r['sigma'] for r in records])
        objective=lambda j:float(np.sum(((predict_at(experiments,j)-values)/sigma)**2))
        optimum=minimize_scalar(objective,bounds=(.8,1.2),method='bounded',options={'xatol':1e-12})
        self.hopping=float(optimum.x)
        return self

    def predict(self,experiments):
        return predict_at(experiments,self.hopping)
