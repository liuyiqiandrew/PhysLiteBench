"""Direct fixed-number boson Hamiltonian, including every physical ring bond."""
from itertools import combinations
from functools import lru_cache
import numpy as np
from scipy.linalg import eigh

LENGTH=8
OCCUPIED={2:(3,4),3:(2,3,4),4:(2,3,4,5)}
TRUE_PARAMETER=1.


def potential(amplitude,asymmetry):
    angle=2*np.pi*np.arange(LENGTH)/LENGTH
    return amplitude*np.cos(angle)+asymmetry*np.sin(angle)


@lru_cache(maxsize=4096)
def state_and_matrix(hopping, number, flux, amplitude, asymmetry, duration):
    states = [sum(1 << j for j in sites) for sites in combinations(range(LENGTH), number)]
    lookup = {bits: index for index, bits in enumerate(states)}
    occupations = np.array([[(bits >> j) & 1 for j in range(LENGTH)] for bits in states])
    matrix = np.diag(occupations @ potential(amplitude, asymmetry)).astype(complex)
    for column, bits in enumerate(states):
        for j in range(LENGTH):
            k = (j + 1) % LENGTH
            for target, origin, phase in [(j, k, 1), (k, j, -1)]:
                if (bits >> origin) & 1 and not (bits >> target) & 1:
                    moved = bits ^ (1 << origin) ^ (1 << target)
                    matrix[lookup[moved], column] += -hopping * np.exp(1j * phase * flux / LENGTH)
    energies, vectors = eigh(matrix)
    initial = lookup[sum(1 << j for j in OCCUPIED[number])]
    state = (vectors * np.exp(-1j * energies * duration)) @ vectors[initial].conj()
    return abs(state)**2@occupations,state,matrix



def predict(experiments,hopping=TRUE_PARAMETER):
    return np.asarray([state_and_matrix(hopping,e['number'],e['flux'],e['amplitude'],e['asymmetry'],e['duration'])[0][e['site']] for e in experiments])


def experiment(number,flux,amplitude,asymmetry,duration,site):
    return dict(number=number,flux=flux,amplitude=amplitude,asymmetry=asymmetry,duration=duration,site=site)


def calibration_inputs():
    return [experiment(3,f,a,b,t,j) for f in [0.,1.,-2.] for a,b in [(0.,0.),(.4,-.3),(.7,.4)] for t in [.12,.2,.3,.4] for j in [1,5]]*4


def hidden_inputs():
    groups={f'number_{n}':[experiment(n,f,a,b,t,j) for f in [0.,1.,-2.] for a,b in [(0.,0.),(.4,-.3),(.7,.4)] for t in [1.5,2.5,3.5] for j in range(8)] for n in [2,4]}
    groups['flux_extrema']=[experiment(n,f,a,b,t,j) for n in [2,4] for f in [-np.pi,np.pi] for a,b in [(0.,0.),(.2,.1)] for t in [1.6,2.4,3.2] for j in range(8)]
    groups['odd_number_anchors']=[experiment(3,f,.53,-.17,t,j) for f in [-2.7,.8,2.5] for t in [.8,2.1,4.] for j in range(8)]
    groups['initial_density']=[experiment(n,1.7,.37,-.21,0.,j) for n in [2,3,4] for j in range(8)]
    return groups
