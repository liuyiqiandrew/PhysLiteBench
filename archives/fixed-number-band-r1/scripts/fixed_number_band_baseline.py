from functools import lru_cache
import numpy as np
from scipy.optimize import brentq, minimize_scalar
from scipy.special import expit


def state(width, temperature, filling):
    energies = np.array([-.5, .5]) * width
    chemical_potential = brentq(
        lambda mu: expit((mu - energies) / temperature).mean() - filling,
        energies[0] - 50*temperature, energies[1] + 50*temperature,
        xtol=5e-15)
    occupations = expit((chemical_potential - energies) / temperature)
    return energies, chemical_potential, occupations


@lru_cache(maxsize=8192)
def capacity(width, temperature, filling):
    energies, chemical_potential, occupations = state(width, temperature, filling)
    weights = occupations * (1 - occupations)
    relative_energy = energies - chemical_potential
    return float(np.mean(relative_energy**2 * weights) / temperature**2)


def predict_at(experiments, width):
    return np.array([capacity(float(width), float(e['temperature']), float(e['filling']))
                     for e in experiments])


class Model:
    def __init__(self):
        self.width = None

    def fit(self, records):
        experiments = [r['input'] for r in records]
        measured = np.array([r['value'] for r in records])
        sigma = np.array([r['sigma'] for r in records])
        def loss(width):
            residual = (predict_at(experiments, width) - measured) / sigma
            return float(residual @ residual)
        result = minimize_scalar(loss, bounds=(.8, 1.2), method='bounded',
                                 options={'xatol': 1e-13})
        self.width = float(min([result.x, .8, 1.2], key=loss))
        return self

    def predict(self, experiments):
        return predict_at(experiments, self.width)
