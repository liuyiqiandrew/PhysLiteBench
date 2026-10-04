import numpy as np
from scipy.optimize import brentq
from functools import lru_cache


@lru_cache(maxsize=4096)
def levels(length, strength, mass, count=16):
    coefficient=1/(2*mass)
    ratio=strength*length/(4*coefficient)
    energies=[]
    for n in range(count):
        lower=(n+.5)*np.pi
        upper=(n+1)*np.pi
        offset=brentq(lambda d:(lower+d)*np.tan(d)-ratio,
                      0.,np.pi/2-1e-12,xtol=2e-14)
        phase=lower+offset
        energies.append(coefficient*(2*phase/length)**2)
        energies.append(coefficient*(2*upper/length)**2)
    return np.sort(energies)


def force(experiment, mass):
    length=experiment['length']
    energy=levels(length,experiment['strength'],mass)
    weight=np.exp(-(energy-energy[0])/experiment['temperature'])
    weight/=weight.sum()
    return float(2*(weight@energy)/length)


def predict_at(experiments, mass):
    return np.array([force(e,mass) for e in experiments])


class Model:
    def __init__(self):
        self.mass=None

    def fit(self, records):
        raise NotImplementedError

    def predict(self, experiments):
        return predict_at(experiments,self.mass)
