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
    hamiltonian = np.diag(energies)
    response = unitary.conj().T@hamiltonian@unitary-hamiltonian
    mean = float(np.trace(response).real/3)
    centered = response-mean*np.eye(3)
    square = centered@centered
    variance = float(np.trace(square).real/3)
    third = float(np.trace(square@centered).real/3)
    fourth = float(np.trace(square@square).real/3)-3*variance**2
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
        raise NotImplementedError

    def predict(self, experiments):
        return predict_at(experiments,self.energy_scale)
