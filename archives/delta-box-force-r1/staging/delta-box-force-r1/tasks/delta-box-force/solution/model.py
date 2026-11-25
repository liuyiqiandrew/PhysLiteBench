import numpy as np
from scipy.optimize import brentq
from functools import lru_cache


@lru_cache(maxsize=4096)
def levels(length, strength, mass, count=16):
    coefficient=1/(2*mass)
    ratio=strength*length/(4*coefficient)
    energies=[]
    forces=[]
    for n in range(count):
        lower=(n+.5)*np.pi
        upper=(n+1)*np.pi
        offset=brentq(lambda d:(lower+d)*np.tan(d)-ratio,
                      0.,np.pi/2-1e-12,xtol=2e-14)
        phase=lower+offset
        wave=2*phase/length
        normalization=length/2-np.sin(wave*length)/(2*wave)
        energies.append(coefficient*wave**2)
        forces.append(coefficient*wave**2/normalization)
        wave=2*upper/length
        energies.append(coefficient*wave**2)
        forces.append(2*coefficient*wave**2/length)
    order=np.argsort(energies)
    return np.array(energies)[order],np.array(forces)[order]


def force(experiment, mass):
    energy,signal=levels(experiment['length'],experiment['strength'],mass)
    weight=np.exp(-(energy-energy[0])/experiment['temperature'])
    weight/=weight.sum()
    return float(weight@signal)


def predict_at(experiments, mass):
    return np.array([force(e,mass) for e in experiments])


class Model:
    def __init__(self):
        self.mass=None

    def fit(self, records):
        from scipy.optimize import minimize_scalar
        experiments=[r['input'] for r in records]
        values=np.array([r['value'] for r in records])
        sigma=np.array([r['sigma'] for r in records])
        def loss(mass):
            residual=(predict_at(experiments,mass)-values)/sigma
            return float(residual@residual)
        self.mass=float(minimize_scalar(loss,bounds=(.4,.9),method='bounded',
            options={'xatol':1e-12}).x)
        return self

    def predict(self, experiments):
        return predict_at(experiments,self.mass)
