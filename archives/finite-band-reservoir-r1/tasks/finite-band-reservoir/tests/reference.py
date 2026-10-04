from functools import lru_cache
import numpy as np
from scipy.special import expit
from scipy.linalg import eigh_tridiagonal
from scipy.fft import dst

TRUE_PARAMETER=.8
PARAMETER='coupling_scale'


def calibration_inputs():
    return [dict(orbital_energy=float(e),temperature=float(t),contact_multiplier=1)
            for repeat in range(32) for e in [.3,.5,.8] for t in [.2,.4,.7]]


def hidden_inputs():
    return {
        'orbital_gate': [dict(orbital_energy=float(e),temperature=.35,contact_multiplier=3) for e in np.linspace(0,.8,10)],
        'bath_temperature': [dict(orbital_energy=.5,temperature=float(t),contact_multiplier=3+i%2) for i,t in enumerate(np.linspace(.2,.8,10))],
        'strong_contact': [dict(orbital_energy=float(e),temperature=float(t),contact_multiplier=4) for e,t in zip(np.linspace(.1,.7,10),np.linspace(.25,.75,10))],
        'weak_contact_anchors': [dict(orbital_energy=float(e),temperature=.55,contact_multiplier=1) for e in np.linspace(.1,.75,8)],
    }

def finite_chain(epsilon,temperature,coupling,length,mu=0.):
    diagonal=np.zeros(length+1);diagonal[0]=epsilon
    hopping=-np.ones(length);hopping[0]=coupling
    E,U=eigh_tridiagonal(diagonal,hopping)
    bath_energies=-2*np.cos(np.pi*np.arange(1,length+1)/(length+1))
    f=expit(-(bath_energies-mu)/temperature)
    projection=dst(U[1:,:],type=1,axis=0,norm='ortho')
    population=U[0]**2+(f[:,None]*projection**2).sum(axis=0)
    value=float((U[0]**2)@population)
    equilibrium=float((U[0]**2)@expit(-(E-mu)/temperature))
    return dict(length=length,correct=value,shortcut=equilibrium,total_particles=float(population.sum()),initial_particles=float(1+f.sum()),energy=float(E@population),initial_energy=float(epsilon+bath_energies@f),eigen_population_min=float(population.min()),eigen_population_max=float(population.max()))


@lru_cache(256)
def _limit(energy,temperature,contact,length):
    rows=[finite_chain(energy,temperature,contact,length*factor)['correct'] for factor in [1,2,4]]
    return rows[0]/3-2*rows[1]+8*rows[2]/3


def predict(experiments, coupling_scale=TRUE_PARAMETER, length=128):
    return np.asarray([_limit(e['orbital_energy'],e['temperature'],coupling_scale*e['contact_multiplier'],length) for e in experiments])
