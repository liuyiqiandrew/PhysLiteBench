from itertools import permutations
from math import factorial
import numpy as np
from scipy.optimize import minimize_scalar


def probability(unitary, gram, counts):
    n = len(counts)
    slots = np.repeat(np.arange(n),counts)
    assignments = list(permutations(range(n)))
    amplitudes = [np.prod(unitary[slots,list(p)]) for p in assignments]
    total = 0j
    for i,p in enumerate(assignments):
        for j,q in enumerate(assignments):
            internal = np.prod([gram[q[k],p[k]] for k in range(n)])
            total += amplitudes[i]*amplitudes[j].conjugate()*internal
    return float(total.real/np.prod([factorial(int(c)) for c in counts]))


class Model:
    def __init__(self):
        self.rotation_gain = None

    def fit(self, records):
        differences = np.array([r['input']['lengths'][1]-r['input']['lengths'][0] for r in records])
        values = np.array([r['value'] for r in records])
        sigma = np.array([r['sigma'] for r in records])
        def loss(gain):
            predicted = .5*np.sin(gain*differences)**2
            return np.sum(((predicted-values)/sigma)**2)
        fit = minimize_scalar(loss,bounds=(.6,1.4),method='bounded',options={'xatol':1e-12})
        self.rotation_gain = float(fit.x)
        return self

    def predict(self, experiments):
        out = []
        for e in experiments:
            n = len(e['lengths'])
            angles = self.rotation_gain*np.array(e['lengths'])
            states = np.column_stack([np.cos(angles),np.sin(angles)])
            gram = states.conj()@states.T
            ports = np.arange(n)
            unitary = np.exp(2j*np.pi*np.outer(ports,ports)/n)/np.sqrt(n)
            out.append(probability(unitary,gram,e['counts']))
        return np.array(out)
