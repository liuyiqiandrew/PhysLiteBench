import numpy as np
from scipy.linalg import eigh
from functools import lru_cache

SITES=8
OCCUPIED=(2,3,4)
POSITIONS=np.arange(SITES)-3.5


def hamiltonian(hopping,trap,tilt):
    matrix=np.diag(trap*POSITIONS**2+tilt*POSITIONS)
    matrix+=np.diag(np.full(SITES-1,-hopping),1)
    matrix+=np.diag(np.full(SITES-1,-hopping),-1)
    return matrix


@lru_cache(maxsize=8192)
def correlation_matrix(hopping,trap,tilt,duration):
    energies,vectors=eigh(hamiltonian(hopping,trap,tilt))
    propagator=(vectors*np.exp(-1j*energies*duration))@vectors.conj().T
    occupied=propagator[:,OCCUPIED]
    return occupied.conj()@occupied.T


def predict_at(experiments,hopping):
    result=[]
    for e in experiments:
        matrix=correlation_matrix(float(hopping),float(e['trap']),float(e['tilt']),float(e['duration']))
        if e['observable']=='density':
            result.append(float(matrix[e['site'],e['site']].real))
        else:
            mode=np.exp(-1j*e['wave_number']*POSITIONS)
            result.append(float(np.vdot(mode,matrix@mode).real/SITES))
    return np.asarray(result)


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
