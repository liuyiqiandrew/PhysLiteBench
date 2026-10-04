"""Equilibrium prediction model for the three-resonator apparatus."""
from functools import lru_cache
import numpy as np

A = np.array([[1.8, .21, -.13], [.21, 2.2, .17], [-.13, .17, 2.6]])
B = np.array([[1.2, .10, .18], [.10, 1.6, -.14], [.18, -.14, 1.7]])


@lru_cache(None)
def normal_modes(pairing):
    pair = pairing*B
    dynamics = np.block([[A, pair], [-pair, -A]])
    values, vectors = np.linalg.eig(dynamics)
    chosen = np.argsort(values.real)[3:]
    energies = values[chosen].real
    modes = vectors[:, chosen]
    weights = abs(modes[:3])**2+abs(modes[3:])**2
    return energies, weights


def predict_at(experiments, energy_scale):
    output = []
    for experiment in experiments:
        energies, weights = normal_modes(experiment['pairing'])
        index = experiment['index']
        if experiment['readout'] == 'frequency':
            value = energy_scale*energies[index]
        elif experiment['readout'] == 'thermal_excess':
            temperature = experiment['temperature']
            if temperature == 0:
                value = 0.
            else:
                population = 1/np.expm1(energy_scale*energies/temperature)
                value = weights[index]@population
        else:
            raise ValueError('unknown readout')
        output.append(float(value))
    return np.asarray(output)


class Model:
    def __init__(self):
        self.energy_scale = None

    def fit(self, records):
        records = list(records)
        unit_response = predict_at([r['input'] for r in records], 1.)
        measured = np.array([r['value'] for r in records])
        sigma = np.array([r['sigma'] for r in records])
        response = unit_response/sigma
        self.energy_scale = float(response@(measured/sigma)/(response@response))
        return self

    def predict(self, experiments):
        if self.energy_scale is None:
            raise RuntimeError('fit must be called before predict')
        return predict_at(experiments, self.energy_scale)
