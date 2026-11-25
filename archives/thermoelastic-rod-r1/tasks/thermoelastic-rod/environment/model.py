from functools import lru_cache
import numpy as np
from scipy.linalg import eigh
from scipy.optimize import minimize_scalar

LENGTH = .05
AREA = 1e-4
VOLUME = LENGTH*AREA
HEAT_FIXED_STRAIN = 1e6
THERMAL_ELASTIC = 300.*2e9*.002**2
HEAT_FIXED_STRESS = HEAT_FIXED_STRAIN+THERMAL_ELASTIC
BATH_CAPACITY = 6.
MODES = 64


@lru_cache(128)
def spectrum(conductivity, contact):
    capacity = np.r_[HEAT_FIXED_STRESS*VOLUME,
                     np.full(MODES, HEAT_FIXED_STRESS*VOLUME), BATH_CAPACITY]
    stiffness = np.diag(np.r_[conductivity*VOLUME*(np.arange(MODES+1)*np.pi/LENGTH)**2, 0.])
    port = np.r_[1., np.full(MODES, np.sqrt(2.)), -1.]
    stiffness += contact*np.outer(port, port)
    rates, vectors = eigh(stiffness/np.sqrt(np.outer(capacity, capacity)))
    return np.maximum(rates, 0), vectors, np.sqrt(capacity)


class Model:
    def __init__(self):
        self.conductivity = None

    def fit(self, records):
        raise NotImplementedError

    def predict(self, experiments):
        out = []
        for e in experiments:
            if e['contact'] == 0:
                values = {'mean': e['mean'], 'bath': e['bath_initial']}
                for n, name in [(1, 'first'), (2, 'second')]:
                    values[name] = e[name]/2*np.exp(-self.conductivity/HEAT_FIXED_STRESS*(n*np.pi/LENGTH)**2*e['time'])
                out.append(values[e['observable']])
                continue
            initial = np.zeros(MODES+2)
            initial[:3] = [e['mean'], e['first']/np.sqrt(2), e['second']/np.sqrt(2)]
            initial[-1] = e['bath_initial']
            rates, vectors, root_capacity = spectrum(self.conductivity, e['contact'])
            state = vectors @ (np.exp(-rates*e['time'])*(vectors.T@(root_capacity*initial)))/root_capacity
            values = {'mean': state[0], 'first': state[1]/np.sqrt(2),
                      'second': state[2]/np.sqrt(2), 'bath': state[-1]}
            out.append(values[e['observable']])
        return np.array(out)
