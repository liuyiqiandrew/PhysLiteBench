import numpy as np
from scipy.linalg import expm


def generator(stage, kappa, chi):
    energy=np.asarray(stage['energies']); barrier=np.asarray(stage['barriers']); affinity=stage['affinity']
    matrix=np.zeros((3,3))
    for i in range(3):
        j=(i+1)%3
        forward=kappa*np.exp(energy[i]-barrier[i]+affinity/6)
        backward=kappa*np.exp(energy[j]-barrier[i]-affinity/6)
        matrix[j,i]+=forward*np.exp(chi/3)
        matrix[i,j]+=backward*np.exp(-chi/3)
        matrix[i,i]-=forward
        matrix[j,j]-=backward
    return matrix


def cycle(e,kappa,chi):
    result=np.eye(3)
    for s in e['stages']:
        result=expm(s['duration']*generator(s,kappa,chi))@result
    return result


def response(e, kappa):
    tilted = cycle(e, kappa, e['field'])
    ordinary = cycle(e, kappa, 0.)
    equation = ordinary-np.eye(3)
    equation[-1, :] = 1.
    p = np.linalg.solve(equation, np.array([0., 0., 1.]))
    return float(np.log(np.sum(tilted@p)))


def predict_at(experiments, attempt_rate):
    return np.asarray([response(e, float(attempt_rate)) for e in experiments])


class Model:
    def __init__(self):
        self.attempt_rate = None

    def fit(self, records):
        basis = predict_at([r['input'] for r in records], 1.)
        sigma = np.asarray([r['sigma'] for r in records])
        value = np.asarray([r['value'] for r in records])
        self.attempt_rate = float(np.sum(basis*value/sigma**2)/np.sum((basis/sigma)**2))
        return self

    def predict(self, experiments):
        return predict_at(experiments, self.attempt_rate)
