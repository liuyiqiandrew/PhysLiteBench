"""Independent conditional oscillator evolution in a truncated Fock basis."""
from functools import lru_cache
import numpy as np
from scipy.linalg import eigh
from scipy.spatial.transform import Rotation

TRUE_PARAMETER = .12
FREQUENCIES = [1.,1.35,1.9,2.6]
STRENGTHS = [.8,.6,.4,.3]
Z = np.array([[1.,1.],[1.,-1.],[-1.,1.],[-1.,-1.]])
PAULI = np.array([[[0,1],[1,0]],[[0,-1j],[1j,0]],[[1,0],[0,-1]]])


def spin_density(rotation, inverse=False):
    phase,angle = rotation
    vector = (-1 if inverse else 1)*angle*np.array([np.cos(phase),np.sin(phase),0.])
    bloch = Rotation.from_rotvec(vector).apply([0.,0.,1.])
    return (np.eye(2)+np.einsum('i,ijk->jk',bloch,PAULI))/2


@lru_cache(4096)
def oscillator(omega, strength, gamma, s, time, cutoff):
    lowering = np.diag(np.sqrt(np.arange(1,cutoff)),1)
    h = omega*np.diag(np.arange(cutoff))+np.sqrt(gamma*strength)*s*(lowering+lowering.T)
    values,vectors = eigh(h)
    return (vectors*np.exp(-1j*values*time))@vectors.T


def reduced_density(e, gamma, cutoff=40):
    state = np.kron(*(spin_density(r) for r in e['preparation']))
    s = Z@np.asarray(e['weights'])
    for omega,strength in zip(FREQUENCIES,STRENGTHS):
        p = np.zeros(cutoff)
        if e['temperature']==0:
            p[0] = 1.
        else:
            p = np.exp(-omega*np.arange(cutoff)/e['temperature'])
            p /= p.sum()
        evolutions = np.array([oscillator(omega,strength,float(gamma),float(v),e['wait'],cutoff) for v in s])
        overlap = np.einsum('aij,bij,j->ab',evolutions,evolutions.conj(),p)
        state *= overlap
    return state


def predict(experiments,gamma,cutoff=40):
    out=[]
    for e in experiments:
        detector=np.kron(*(spin_density(r,True) for r in e['readout']))
        out.append(float(np.real(np.trace(detector@reduced_density(e,gamma,cutoff)))))
    return np.array(out)


def experiment(wait,weights,temperature=.2,preparation=None,readout=None):
    return dict(wait=float(wait),weights=list(weights),temperature=float(temperature),
                preparation=preparation if preparation is not None else [[np.pi/2,np.pi/2],[np.pi/2,np.pi/2]],
                readout=readout if readout is not None else [[np.pi/2,-np.pi/2],[np.pi/2,-np.pi/2]])


def calibration_inputs():
    out=[]
    for active in range(2):
        for temperature in [0.,.2,.5,.8]:
            for weight in [.7,1.1]:
                for time in [.3,.7,1.2,2.,3.,4.,6.,8.]:
                    w=[0.,0.];w[active]=weight
                    prep=[[0.,0.],[0.,0.]];read=[[0.,0.],[0.,0.]]
                    prep[active]=[np.pi/2,np.pi/2];read[active]=[np.pi/2,-np.pi/2]
                    out.append(experiment(time,w,temperature,prep,read))
    return out


def hidden_inputs():
    return {
        'paired_preparations':[experiment(t,[1.,1.],temperature) for t in [1.2,2.5,4.,6.,8.5] for temperature in [0.,.35]],
        'signed_couplings':[experiment(t,w,.15) for t in [2.,4.,7.,9.] for w in [[1.,-.8],[-1.1,-.7],[.8,1.2]]],
        'rotated_readout':[experiment(t,[.9,1.1],.25,[[.2,1.1],[-.6,1.7]],[[.9,-1.4],[-.4,.8]]) for t in [1.,2.5,4.,6.,8.,10.]],
    }
