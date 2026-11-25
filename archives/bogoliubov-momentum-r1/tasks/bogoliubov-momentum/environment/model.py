import numpy as np
from numpy.polynomial.legendre import leggauss
from functools import lru_cache


@lru_cache(None)
def quadrature(order):
    nodes,weights = leggauss(order)
    return (nodes+1)/2,weights/2


def mode_statistics(momentum, temperature, interaction):
    kinetic = momentum**2/2
    energy = np.sqrt(kinetic*(kinetic+2*interaction))
    if temperature == 0:
        population = np.zeros_like(momentum)
    else:
        decay = np.exp(-energy/temperature)
        population = decay/(-np.expm1(-energy/temperature))
    vacuum = interaction**2/(2*energy*(kinetic+interaction+energy))
    return population,1+2*vacuum,vacuum


@lru_cache(None)
def depletion(temperature, interaction, order=192):
    nodes,weights = quadrature(order)
    scale = np.sqrt(2*(temperature+interaction)) if temperature+interaction>0 else 1.
    momentum = scale*nodes/(1-nodes)
    jacobian = scale/(1-nodes)**2
    population,mixing,vacuum = mode_statistics(momentum,temperature,interaction)
    return float(weights@(jacobian*momentum**2*(mixing*population+vacuum)))/(2*np.pi**2)


@lru_cache(None)
def variance_density(temperature, interaction, order=192):
    return temperature*(depletion(temperature,interaction,order)-depletion(0.,interaction,order))


def predict_at(experiments, gain):
    return gain*np.array([variance_density(e['temperature'],e['interaction']) for e in experiments])


class Model:
    def __init__(self):
        self.variance_gain = None

    def fit(self, records):
        raise NotImplementedError

    def predict(self, experiments):
        return predict_at(experiments,self.variance_gain)
