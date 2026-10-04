import numpy as np
from scipy.linalg import eigh
from scipy.optimize import minimize_scalar
from functools import lru_cache
from itertools import combinations

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


@lru_cache(maxsize=None)
def _basis(number):
    """Occupation-number basis for ``number`` hard-core bosons."""
    states=tuple(combinations(range(SITES),number))
    indices={state:index for index,state in enumerate(states)}
    return states,indices


@lru_cache(maxsize=16384)
def _many_body_hamiltonian(hopping,number,flux,amplitude,asymmetry):
    """Construct the exact hard-core-boson Hamiltonian in the Fock basis."""
    states,indices=_basis(number)
    matrix=np.zeros((len(states),len(states)),dtype=complex)
    site_potential=potential(amplitude,asymmetry)

    for column,state in enumerate(states):
        occupied=set(state)
        matrix[column,column]=sum(site_potential[site] for site in state)

        # Apply each directed hopping term to this occupation state.  Unlike
        # a fermionic basis, hard-core boson hops have no exchange sign.
        for source in state:
            for destination in ((source-1)%SITES,(source+1)%SITES):
                if destination in occupied:
                    continue
                new_state=tuple(sorted((occupied-{source})|{destination}))
                row=indices[new_state]
                if destination==(source-1)%SITES:
                    phase=np.exp(1j*flux/SITES)
                else:
                    phase=np.exp(-1j*flux/SITES)
                matrix[row,column]=-hopping*phase

    return matrix


@lru_cache(maxsize=16384)
def density(hopping,number,flux,amplitude,asymmetry,duration):
    states,indices=_basis(number)
    energies,vectors=eigh(
        _many_body_hamiltonian(hopping,number,flux,amplitude,asymmetry)
    )
    initial=indices[tuple(OCCUPIED[number])]
    evolved=vectors@(
        np.exp(-1j*energies*duration)*vectors[initial,:].conj()
    )
    probabilities=np.abs(evolved)**2

    occupations=np.zeros((SITES,len(states)))
    for state_index,state in enumerate(states):
        occupations[list(state),state_index]=1.0
    return occupations@probabilities


def predict_at(experiments,hopping):
    return np.asarray([
        density(
            float(hopping),
            int(e['number']),
            float(e['flux']),
            float(e['amplitude']),
            float(e['asymmetry']),
            float(e['duration']),
        )[int(e['site'])]
        for e in experiments
    ],dtype=float)


class Model:
    def __init__(self):
        self.hopping=None

    def fit(self,records):
        records=list(records)
        if not records:
            raise ValueError('At least one calibration record is required.')

        experiments=[record['input'] for record in records]
        values=np.asarray([float(record['value']) for record in records])
        sigmas=np.asarray([float(record['sigma']) for record in records])
        if np.any(sigmas<=0) or not np.isfinite(sigmas).all():
            raise ValueError('Calibration uncertainties must be finite and positive.')

        def objective(hopping):
            residual=(predict_at(experiments,hopping)-values)/sigmas
            return float(np.dot(residual,residual))

        result=minimize_scalar(
            objective,
            bounds=(0.8,1.2),
            method='bounded',
            options={'xatol':1e-12},
        )
        if not result.success or not np.isfinite(result.x):
            raise RuntimeError('Unable to fit the hopping parameter.')
        self.hopping=float(np.clip(result.x,0.8,1.2))
        return self

    def predict(self,experiments):
        return predict_at(experiments,self.hopping)
