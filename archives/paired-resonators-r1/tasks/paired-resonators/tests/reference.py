"""Independent mass-coordinate quantization of the real quadratic network."""
from functools import lru_cache
import numpy as np
from scipy.linalg import eigh

TRUE_PARAMETER = 1.04


@lru_cache(None)
def quadrature_modes(pairing):
    potential = np.array([[1.8+1.2*pairing, .21+.10*pairing, -.13+.18*pairing],
                          [.21+.10*pairing, 2.2+1.6*pairing, .17-.14*pairing],
                          [-.13+.18*pairing, .17-.14*pairing, 2.6+1.7*pairing]])
    kinetic = np.array([[1.8-1.2*pairing, .21-.10*pairing, -.13-.18*pairing],
                        [.21-.10*pairing, 2.2-1.6*pairing, .17+.14*pairing],
                        [-.13-.18*pairing, .17+.14*pairing, 2.6-1.7*pairing]])
    values, vectors = eigh(kinetic)
    root = (vectors*np.sqrt(values))@vectors.T
    inverse_root = (vectors/np.sqrt(values))@vectors.T
    squared_frequencies, rotation = eigh(root@potential@root)
    return np.sqrt(squared_frequencies), root@rotation, inverse_root@rotation


def predict(experiments, energy_scale=TRUE_PARAMETER):
    output = []
    for e in experiments:
        frequencies, position_map, momentum_map = quadrature_modes(e['pairing'])
        if e['readout'] == 'frequency':
            value = energy_scale*frequencies[e['index']]
        elif e['temperature'] == 0:
            value = 0.
        else:
            population = 1/np.expm1(energy_scale*frequencies/e['temperature'])
            q_variance = (position_map**2)@(population/frequencies)
            p_variance = (momentum_map**2)@(population*frequencies)
            value = (q_variance[e['index']]+p_variance[e['index']])/2
        output.append(value)
    return np.asarray(output)


def experiment(readout='thermal_excess', index=0, pairing=.8, temperature=.5):
    return dict(readout=readout, index=index, pairing=pairing, temperature=temperature)


def calibration_inputs():
    return [experiment('frequency', j, s, .35) for s in [0.,.2,.4,.6,.8,1.] for j in range(3)]


def hidden_inputs():
    return {
        'strong_pairing': [experiment(index=j,pairing=s,temperature=t)
                           for s,t in [(.9,.25),(1.,.45),(.95,.75)] for j in range(3)],
        'moderate_pairing': [experiment(index=j,pairing=s,temperature=t)
                             for s,t in [(.55,.4),(.7,.6),(.8,.8)] for j in range(3)],
        'temperature_scan': [experiment(index=j,pairing=.98,temperature=t)
                              for t in [.15,.3,.55,.8] for j in range(3)],
        'spectroscopy_and_unpaired': [experiment('frequency',j,s,.5)
                                      for s in [.1,.65,.97] for j in range(3)]
                                     +[experiment(index=j,pairing=0.,temperature=.65) for j in range(3)]
                                     +[experiment(index=j,pairing=1.,temperature=0.) for j in range(3)],
    }
