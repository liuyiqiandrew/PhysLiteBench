from functools import lru_cache
import numpy as np
from scipy.linalg import eigh

TRUE_PARAMETER = .014


def experiment(frequency, interaction, temperature):
    return dict(frequency=float(frequency), interaction=float(interaction), temperature=float(temperature))


def calibration_inputs():
    return [experiment(w, 0., t) for _ in range(8) for w in [1.1, 1.45, 1.8]
            for t in [.16, .2, .26, .34, .44, .55]]


def hidden_inputs():
    return {
        'interaction_sweep': [experiment(1.3, ratio*np.sqrt(1.3), .3) for ratio in [.28, .32, .36]],
        'temperature_sweep': [experiment(1.4, .37*np.sqrt(1.4), t) for t in [.18, .3, .45]],
        'frequency_sweep': [experiment(w, .37*np.sqrt(w), .4) for w in [1.1, 1.5, 1.8]],
    }


@lru_cache(maxsize=64)
def spectrum(frequency, interaction, cutoff=36):
    a = np.diag(np.sqrt(np.arange(1, cutoff)), 1)
    identity = np.eye(cutoff)
    first = np.kron(a, identity)
    second = np.kron(identity, a)
    quadrature = first+first.T
    number = first.T@first
    hamiltonian = number+frequency*(second.T@second)+interaction*quadrature@(second+second.T)
    energies, vectors = eigh(hamiltonian)
    transition = vectors.T@quadrature@vectors
    losses = energies[None, :]-energies[:, None]
    weights = np.sum(transition**2*((losses>=.2)&(losses<=2.5)), axis=0)
    bare = np.diag(vectors.T@number@vectors).copy()
    return energies-energies[0], weights, bare


def fock_response(experiment, cutoff=36):
    energy, weights, _ = spectrum(experiment['frequency'], experiment['interaction'], cutoff)
    population = np.exp(-energy/experiment['temperature'])
    population /= population.sum()
    return float(population@weights)


def predict(experiments, detection_rate):
    return detection_rate*np.array([1/np.expm1(1/e['temperature']) if e['interaction']==0
                                    else fock_response(e) for e in experiments])
