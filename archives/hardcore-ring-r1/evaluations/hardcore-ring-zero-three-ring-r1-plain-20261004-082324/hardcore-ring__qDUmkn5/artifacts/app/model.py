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
    """Return the fixed-number hard-core Fock basis as bit masks."""
    return tuple(sum(1 << site for site in sites)
                 for sites in combinations(range(SITES), number))


@lru_cache(maxsize=None)
def _basis_index(number):
    return {state: index for index, state in enumerate(_basis(number))}


@lru_cache(maxsize=16384)
def many_body_hamiltonian(hopping,number,flux,amplitude,asymmetry):
    """Build the exact hard-core boson Hamiltonian in the number sector.

    A bit in a basis mask denotes an occupied site.  The matrix elements have
    no fermionic exchange signs: the particles are hard-core bosons, as in the
    Hamiltonian in the problem statement.
    """
    number = int(number)
    basis = _basis(number)
    index = _basis_index(number)
    matrix = np.zeros((len(basis), len(basis)), dtype=complex)
    onsite = potential(amplitude, asymmetry)
    phase = np.exp(1j * flux / SITES)

    for column, state in enumerate(basis):
        matrix[column, column] = sum(onsite[site]
                                     for site in range(SITES)
                                     if state & (1 << site))
        for site in range(SITES):
            neighbour = (site + 1) % SITES

            # b_site^dagger b_neighbour: move a particle neighbour -> site.
            if (state & (1 << neighbour)) and not (state & (1 << site)):
                target = state ^ (1 << neighbour) ^ (1 << site)
                matrix[index[target], column] += -hopping * phase

            # b_neighbour^dagger b_site: move a particle site -> neighbour.
            if (state & (1 << site)) and not (state & (1 << neighbour)):
                target = state ^ (1 << site) ^ (1 << neighbour)
                matrix[index[target], column] += -hopping * phase.conjugate()

    return matrix


@lru_cache(maxsize=16384)
def density(hopping,number,flux,amplitude,asymmetry,duration):
    """Return all site occupations after the specified evolution."""
    number = int(number)
    basis = _basis(number)
    initial_state = sum(1 << site for site in OCCUPIED[number])
    initial_index = _basis_index(number)[initial_state]

    energies, vectors = eigh(
        many_body_hamiltonian(hopping, number, flux, amplitude, asymmetry)
    )
    coefficients = vectors.conj().T[:, initial_index]
    state = vectors @ (np.exp(-1j * energies * duration) * coefficients)

    occupations = np.zeros(SITES, dtype=float)
    probabilities = np.abs(state) ** 2
    for index, basis_state in enumerate(basis):
        for site in range(SITES):
            if basis_state & (1 << site):
                occupations[site] += probabilities[index]
    return occupations


def predict_at(experiments,hopping):
    if hopping is None:
        raise ValueError('Model must be fitted before prediction.')

    return np.asarray([
        density(float(hopping), int(experiment['number']),
                float(experiment['flux']), float(experiment['amplitude']),
                float(experiment['asymmetry']), float(experiment['duration']))[
                    int(experiment['site'])]
        for experiment in experiments
    ], dtype=float)


class Model:
    def __init__(self):
        self.hopping=None

    def fit(self,records):
        records = list(records)
        if not records:
            raise ValueError('At least one calibration record is required.')

        experiments = [record['input'] for record in records]
        values = np.asarray([record['value'] for record in records], dtype=float)
        sigmas = np.asarray([record['sigma'] for record in records], dtype=float)
        if (not np.all(np.isfinite(values)) or
                not np.all(np.isfinite(sigmas)) or np.any(sigmas <= 0)):
            raise ValueError('Calibration values and uncertainties must be finite, with sigma > 0.')

        def objective(hopping):
            residual = (predict_at(experiments, hopping) - values) / sigmas
            return float(np.dot(residual, residual))

        result = minimize_scalar(objective, bounds=(0.8, 1.2), method='bounded',
                                 options={'xatol': 1e-12})
        if not result.success or not np.isfinite(result.x):
            raise RuntimeError('Unable to fit the hopping from the calibration records.')
        self.hopping = float(result.x)
        return self

    def predict(self,experiments):
        return predict_at(experiments,self.hopping)
