import numpy as np
from scipy.linalg import expm
from scipy.special import expit

STATES = [(0,0),(1,0),(0,1),(1,1)]


def dynamics(bias, rate):
    filling = [expit(bias/2),expit(-bias/2)]
    edges = []
    for j,state in enumerate(STATES):
        for site,bond in [(0,0),(1,2)]:
            target = list(state)
            target[site] = 1-target[site]
            jump = (1-2*state[site])*(1 if site == 0 else -1)
            value = rate*(filling[site] if state[site] == 0 else 1-filling[site])
            edges.append((STATES.index(tuple(target)),j,value,bond,jump))
        if state[0] != state[1]:
            edges.append((STATES.index(state[::-1]),j,rate,1,state[0]-state[1]))
    generator = np.zeros((4,4))
    for i,j,value,bond,jump in edges:
        generator[i,j] += value
        generator[j,j] -= value
    normalized = generator.copy()
    normalized[-1,:] = 1.
    probability = np.linalg.solve(normalized,[0.,0.,0.,1.])
    return generator,edges,probability


def count_variance(generator, edges, probability, duration, selected_bond):
    first = np.zeros_like(generator)
    second = np.zeros_like(generator)
    for i,j,value,bond,jump in edges:
        increment = jump if bond == selected_bond else 0.
        first[i,j] += value*increment
        second[i,j] += value*increment**2
    zero = np.zeros_like(generator)
    augmented = np.block([[generator,zero,zero],
                          [first,generator,zero],
                          [second,2*first,generator]])
    result = expm(augmented*duration)@np.r_[probability,np.zeros(8)]
    mean = np.sum(result[4:8])
    return float(np.sum(result[8:])-mean**2)


class Model:
    def __init__(self):
        self.rate = None

    def fit(self, records):
        coefficient = np.array([r['input']['duration']*np.tanh(r['input']['bias']/4)/3 for r in records])
        values = np.array([r['value'] for r in records])
        sigma = np.array([r['sigma'] for r in records])
        self.rate = float(np.sum(coefficient*values/sigma**2)/np.sum(coefficient**2/sigma**2))
        return self

    def predict(self, experiments):
        out = []
        for e in experiments:
            if e['statistic'] == 'mean':
                value = self.rate*e['duration']*np.tanh(e['bias']/4)/3*sum(e['weights'])
            else:
                generator,edges,probability = dynamics(e['bias'],self.rate)
                value = sum(weight**2*count_variance(generator,edges,probability,e['duration'],bond)
                            for bond,weight in enumerate(e['weights']) if weight != 0)
            out.append(value)
        return np.array(out)
