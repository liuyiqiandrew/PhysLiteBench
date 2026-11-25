"""Atomic-mode Gaussian contractions, independent adaptive momentum integration."""
from functools import lru_cache
import json
from pathlib import Path
import numpy as np
from scipy.integrate import quad

TRUE_PARAMETER=1.08


def experiment(temperature,interaction):
    return dict(temperature=temperature,interaction=interaction)


def atomic_covariances(momentum,temperature,interaction):
    free_energy=momentum*momentum/2
    energy=np.sqrt(free_energy*(free_energy+2*interaction))
    # Two-mode canonical Gaussian covariance in the physical particle basis.
    thermal_factor=1. if temperature==0 else 1/np.tanh(energy/(2*temperature))
    normal=.5*((free_energy+interaction)*thermal_factor/energy-1)
    anomalous=-.5*interaction*thermal_factor/energy
    return normal,anomalous


@lru_cache(None)
def unit_variance(temperature,interaction):
    if temperature==0:
        return 0.
    def integrand(momentum):
        normal,anomalous=atomic_covariances(momentum,temperature,interaction)
        # Opposite momenta enter P with opposite signs. The full momentum
        # integral counts both members of each pair once.
        variance=normal*(normal+1)-anomalous*anomalous
        return momentum**4*variance/(6*np.pi**2)
    upper=np.sqrt(2*(48*temperature+interaction))
    return quad(integrand,0,upper,epsabs=1e-13,epsrel=2e-10,limit=120)[0]


def predict(experiments,gain):
    return gain*np.array([unit_variance(e['temperature'],e['interaction']) for e in experiments])


def hidden_inputs():
    return {
        'cold_interacting':[experiment(t,u) for t,u in [(.08,.5),(.12,.8),(.2,1.),(.18,.5)]],
        'mixed_spectrum':[experiment(t,u) for t,u in [(.25,.25),(.4,.6),(.6,.7),(.8,1.)]],
        'weak_repulsion':[experiment(t,u) for t,u in [(.2,.12),(.35,.15),(.6,.2),(.9,.3)]],
        'ideal_and_ground':[experiment(.15,0.),experiment(.65,0.),experiment(0.,.4),experiment(0.,1.)]
    }


def calibration_inputs():
    return [experiment(t,0.) for t in [.12,.2,.35,.55,.75,.9]]*24


def generate_data():
    inputs=calibration_inputs();values=predict(inputs,TRUE_PARAMETER);rng=np.random.default_rng(456831)
    records=[dict(input=e,value=float(v+1e-5*z),sigma=1e-5) for e,v,z in zip(inputs,values,rng.normal(size=len(inputs)))]
    task=Path(__file__).resolve().parents[1]
    for filename in ['environment/data/calibration.json','tests/data/calibration.json']:(task/filename).write_text(json.dumps(records,indent=2)+'\n')
