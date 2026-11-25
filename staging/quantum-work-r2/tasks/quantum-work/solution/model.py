import numpy as np
from scipy.linalg import expm
from functools import lru_cache

D = np.diag([0.,1.,2.35])
X = np.array([[0.,1.,.25],[1.,0.,.8],[.25,.8,0.]])
Y = np.array([[0.,-1j,.45j],[1j,0.,-.7j],[-.45j,.7j,0.]])


@lru_cache(maxsize=4096)
def evolution(scale, amplitude_a, amplitude_b, phase, time_a, time_b):
    initial = scale*D
    first = initial+amplitude_a*X
    second = initial+amplitude_b*(np.cos(phase)*X+np.sin(phase)*Y)
    return expm(-1j*time_b*second)@expm(-1j*time_a*first)


def statistics(amplitude_a, amplitude_b, phase, time_a, time_b, scale):
    energies = scale*np.diag(D)
    unitary = evolution(scale,amplitude_a,amplitude_b,phase,time_a,time_b)
    probability = abs(unitary)**2/3
    change = energies[:,None]-energies[None,:]
    mean = float(np.sum(probability*change))
    centered = change-mean
    variance = float(np.sum(probability*centered**2))
    third = float(np.sum(probability*centered**3))
    fourth = float(np.sum(probability*centered**4))-3*variance**2
    return np.array([mean,variance,third,fourth])


def predict_at(experiments, scale):
    keys = ['amplitude_a','amplitude_b','phase','time_a','time_b']
    cache = {}
    values = []
    for experiment in experiments:
        key = tuple(experiment[k] for k in keys)
        if key not in cache:
            cache[key] = statistics(*key,scale)
        values.append(cache[key][experiment['cumulant']-1])
    return np.array(values)


class Model:
    def __init__(self):
        self.energy_scale = None

    def fit(self, records):
        from scipy.optimize import minimize_scalar
        experiments = [r['input'] for r in records]
        values = np.array([r['value'] for r in records])
        sigma = np.array([r['sigma'] for r in records])
        def loss(scale):
            residual = (predict_at(experiments,scale)-values)/sigma
            return float(residual@residual)
        result = minimize_scalar(loss,bounds=(.8,1.2),method='bounded',
                                 options={'xatol':1e-12})
        choices = [(result.fun,result.x),(loss(.8),.8),(loss(1.2),1.2)]
        self.energy_scale = float(min(choices)[1])
        return self

    def predict(self, experiments):
        return predict_at(experiments,self.energy_scale)
