"""Finite-gap lab-spin Fourier Hamiltonian and kinetic-energy thermal trace."""
from functools import lru_cache
import numpy as np

TRUE_PARAMETER=1.07


def experiment(temperature=.1,winding=1,tilt=np.pi/2):
    return dict(temperature=float(temperature),winding=int(winding),tilt=float(tilt))


def finite_gap(inertia,temperature,winding,tilt,gap,cutoff=24):
    n=np.arange(-cutoff,cutoff+1,dtype=float)
    kinetic=np.column_stack(((n-winding)**2/(2*inertia),n**2/(2*inertia)))
    h=np.zeros((len(n),2,2))
    h[:,0,0]=kinetic[:,0]+gap*np.cos(tilt)
    h[:,1,1]=kinetic[:,1]-gap*np.cos(tilt)
    h[:,0,1]=h[:,1,0]=gap*np.sin(tilt)
    energies,vectors=np.linalg.eigh(h)
    measured=np.einsum('nai,na,nai->ni',vectors,kinetic,vectors)
    population=np.exp(-(energies-energies.min())/temperature)
    population/=population.sum()
    return float(np.sum(population*measured))


@lru_cache(maxsize=4096)
def kinetic_energy(inertia,temperature,winding,tilt,base=1024.,cutoff=24):
    values=[finite_gap(inertia,temperature,winding,tilt,base*f,cutoff) for f in [1,2,4]]
    return float(np.array(values)@np.array([1/3,-2,8/3]))


def predict(experiments,inertia=TRUE_PARAMETER):
    return np.array([kinetic_energy(inertia,e['temperature'],e['winding'],e['tilt']) for e in experiments])


def calibration_inputs():
    return [experiment(t,q,np.pi/2) for _ in range(36) for t in [.06,.1,.18,.3] for q in [2,4]]


def hidden_inputs():
    return {
        'odd_winding':[experiment(t,1,np.pi/2) for t in [.05,.08,.12,.18]],
        'tilted_double_winding':[experiment(t,2,np.pi/3) for t in [.05,.08,.12,.18]],
        'tilted_single_winding':[experiment(t,1,.65*np.pi) for t in [.05,.08,.12,.18]],
        'even_winding_anchors':[experiment(.04,2,np.pi/2),experiment(.35,4,np.pi/2)],
    }
