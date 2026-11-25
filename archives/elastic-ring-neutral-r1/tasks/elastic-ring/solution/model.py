import numpy as np
from numpy.polynomial.legendre import leggauss

NODES, WEIGHTS = leggauss(128)


def predict_at(experiments, stiffness):
    values = []
    for experiment in experiments:
        preferred = experiment['preferred']
        readout = experiment['readout']
        if readout == 'torque':
            values.append(-stiffness*np.sin(experiment['angle']-preferred))
            continue
        cutoff = experiment['cutoff']
        angles = cutoff+(NODES+1)*(np.pi-2*cutoff)/2
        potential = stiffness*(1-np.cos(angles-preferred))
        measure = 1/np.sin(angles)
        weight = WEIGHTS*measure*np.exp(-potential/experiment['temperature'])
        observable = np.sin(angles) if readout == 'sine' else np.cos(2*angles)
        values.append(weight@observable/weight.sum())
    return np.array(values)


class Model:
    def __init__(self):
        self.stiffness = None

    def fit(self, records):
        basis = np.array([-np.sin(r['input']['angle']-r['input']['preferred']) for r in records])
        values = np.array([r['value'] for r in records])
        variance = np.array([r['sigma']**2 for r in records])
        self.stiffness = float(np.sum(basis*values/variance)/np.sum(basis*basis/variance))
        return self

    def predict(self, experiments):
        return predict_at(experiments, self.stiffness)
