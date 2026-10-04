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


def reduced_density(e, gamma, cutoff=56):
    state = np.kron(*(spin_density(r) for r in e['preparation']))
    # Independent spin pulse via the matrix exponential of its generator.
    from scipy.linalg import expm
    def pulse(r):
        phase,angle=r
        return expm(-.5j*angle*(np.cos(phase)*PAULI[0]+np.sin(phase)*PAULI[1]))
    control=np.kron(*(pulse(r) for r in e['intermediate']))
    final,initial=np.indices((4,4));first,second=initial.ravel(),final.ravel()
    s=Z@np.asarray(e['weights']);overlap=np.ones((16,16),complex)
    for omega,strength in zip(FREQUENCIES,STRENGTHS):
        p=np.zeros(cutoff)
        if e['temperature']==0:p[0]=1.
        else:
            p=np.exp(-omega*np.arange(cutoff)/e['temperature']);p/=p.sum()
        u1=[oscillator(omega,strength,float(gamma),float(v),e['waits'][0],cutoff) for v in s]
        u2=[oscillator(omega,strength,float(gamma),float(v),e['waits'][1],cutoff) for v in s]
        paths=np.array([u2[j]@u1[i] for i,j in zip(first,second)])
        overlap*=np.einsum('aij,bij,j->ab',paths,paths.conj(),p)
    coefficients=state[first[:,None],first[None,:]]*control[second,first][:,None]*control[second,first].conj()[None,:]
    reduced=np.zeros((4,4),complex)
    np.add.at(reduced,(second[:,None],second[None,:]),coefficients*overlap)
    return reduced


def predict(experiments,gamma,cutoff=56):
    out=[]
    for e in experiments:
        detector=np.kron(*(spin_density(r,True) for r in e['readout']))
        out.append(float(np.real(np.trace(detector@reduced_density(e,gamma,cutoff)))))
    return np.array(out)


def experiment(waits,weights,temperature=.2,preparation=None,readout=None,intermediate=None):
    return dict(waits=list(waits),weights=list(weights),temperature=float(temperature),
                preparation=preparation if preparation is not None else [[np.pi/2,np.pi/2],[np.pi/2,np.pi/2]],
                intermediate=intermediate if intermediate is not None else [[0.,0.],[0.,0.]],
                readout=readout if readout is not None else [[np.pi/2,-np.pi/2],[np.pi/2,-np.pi/2]])


def calibration_inputs():
    return [experiment([time,0.],weights,temperature)
            for weights in [[.9,0.],[0.,1.1],[.7,.9],[1.1,-.8]]
            for temperature in [0.,.2,.5,.8]
            for time in [.3,.7,1.2,2.,3.,4.,6.,8.]]


def hidden_inputs():
    return {
        'echo_sequences':[experiment([t1,t2],[1.,1.],temp,intermediate=[[0.,np.pi],[0.,0.]])
                          for t1,t2 in [(1.,2.),(2.,2.5),(3.,1.),(4.,4.)] for temp in [0.,.4]],
        'rotated_sequences':[experiment([t1,t2],[.9,-1.1],.15,
                             [[.2,1.1],[-.6,1.7]],[[.9,-1.4],[-.4,.8]],[[.6,1.4],[-.3,-1.1]])
                             for t1,t2 in [(.7,2.5),(1.5,3.),(2.,4.),(3.,2.),(4.5,1.5),(4.,4.)]],
        'mixed_sequences':[experiment([t1,t2],w,.25,intermediate=[[np.pi/2,.8],[0.,-2.2]])
                           for t1,t2 in [(1.2,2.4),(2.5,3.5),(4.,2.)] for w in [[1.1,.7],[-.8,1.2]]],
    }
