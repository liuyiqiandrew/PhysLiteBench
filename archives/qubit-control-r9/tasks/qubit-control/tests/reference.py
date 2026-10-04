"""Full interacting Gibbs preparation and evolution in finite Fock spaces."""
from functools import lru_cache
import numpy as np
from scipy.linalg import eigh, expm
from scipy.special import logsumexp, softmax

TRUE_PARAMETER = .12
FREQUENCIES = [1.,1.35,1.9,2.6]
STRENGTHS = [.8,.6,.4,.3]
Z = np.array([[1.,1.],[1.,-1.],[-1.,1.],[-1.,-1.]])
SX = np.array([[0.,1.],[1.,0.]])
SY = np.array([[0.,-1j],[1j,0.]])


def pulse(r):
    phase,angle=r
    return expm(-.5j*angle*(np.cos(phase)*SX+np.sin(phase)*SY))


@lru_cache(512)
def oscillator(omega, strength, gamma, s, cutoff):
    lowering = np.diag(np.sqrt(np.arange(1,cutoff)),1)
    h = omega*np.diag(np.arange(cutoff))+np.sqrt(gamma*strength)*s*(lowering+lowering.T)
    return eigh(h)


def reduced_density(e,gamma,cutoff=64):
    s=Z@np.asarray(e['weights']);log_weights=np.zeros(4)
    overlaps=np.ones((4,4,4),complex)
    for omega,strength in zip(FREQUENCIES,STRENGTHS):
        energies=[];vectors=[];propagators=[]
        for r,force in enumerate(s):
            energy,vector=oscillator(omega,strength,float(gamma),float(force),cutoff)
            energies.append(energy);vectors.append(vector)
            log_weights[r]+=logsumexp(-energy/e['temperature'])
            propagators.append((vector*np.exp(-1j*energy*e['wait']))@vector.T)
        propagators=np.array(propagators)
        for r in range(4):
            population=softmax(-energies[r]/e['temperature'])
            conditional=(vectors[r]*population)@vectors[r].T
            overlaps[r]*=np.einsum('aij,jk,bik->ab',propagators,conditional,propagators.conj(),optimize=True)
    weights=softmax(log_weights)
    preparation=np.kron(*(pulse(r) for r in e['preparation']))
    density=np.zeros((4,4),complex)
    for r in range(4):
        state=np.outer(preparation[:,r],preparation[:,r].conj())
        density+=weights[r]*state*overlaps[r]
    return density


def predict(experiments,gamma,cutoff=64):
    out=[]
    for e in experiments:
        readout=np.kron(*(pulse(r) for r in e['readout']))
        density=readout@reduced_density(e,gamma,cutoff)@readout.conj().T
        out.append(float(density[0,0].real))
    return np.array(out)


def experiment(wait,weights,temperature=.4,preparation=None,readout=None):
    return dict(wait=float(wait),weights=list(weights),temperature=float(temperature),
                preparation=preparation if preparation is not None else [[0.,0.],[0.,0.]],
                readout=readout if readout is not None else [[0.,0.],[0.,0.]])


def calibration_inputs():
    return [experiment(time,weights,temperature,readout=readout)
            for _ in range(2)
            for weights in [[.5,.7],[.8,1.],[1.1,-.6],[-.9,1.2]]
            for temperature in [.35,.55,.85,1.2]
            for time in [0.,.8,3.]
            for readout in [[[0.,0.],[0.,0.]],[[0.,np.pi],[0.,0.]],[[.2,np.pi/3],[-.4,np.pi/4]]]]


def hidden_inputs():
    return {
        'joint_quadrature':[experiment(t,[1.2,1.2],T,[[0.,np.pi/2],[0.,np.pi/2]],
                                    [[np.pi/2,np.pi/2],[np.pi/2,np.pi/2]])
                            for t in [.3,.5,.7,1.,1.4,2.] for T in [.25,.5]],
        'signed_weights':[experiment(t,[1.2,-1.],T,[[0.,1.4],[0.,1.7]],
                                    [[1.3,1.4],[1.7,1.6]])
                          for t in [.25,.45,.7,1.,1.5,2.] for T in [.3,.65]],
        'conditional_rotation':[experiment(t,[1.1,.9],T,[[np.pi/2,np.pi/2],[0.,0.]],
                                         [[0.,np.pi/2],[0.,0.]])
                                for t in [.25,.45,.7,1.,1.4,1.8] for T in [.25,.5]],
    }
