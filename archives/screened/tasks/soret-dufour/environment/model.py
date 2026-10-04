import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize_scalar

CAPACITY = 2e6
CONDUCTIVITY = .2
DENSITY = 1000.
T0 = 300.
C0 = .5
SORET = .05
GAS_CONSTANT = 8.314462618


class Model:
    def __init__(self):
        self.diffusivity = None

    def fit(self, records):
        raise NotImplementedError

    def predict(self, experiments):
        out = []
        d = self.diffusivity
        transport = np.array([[CONDUCTIVITY*1e6/CAPACITY,0.],
                              [d*C0*(1-C0)*SORET,d]])
        for e in experiments:
            q2 = (e['mode']*np.pi/e['length'])**2
            if e['isothermal']:
                state = np.array([0.,e['composition_amplitude']*np.exp(-d*q2*e['time'])])
            else:
                initial = np.array([e['temperature_amplitude'],e['composition_amplitude']])
                state = expm(-q2*e['time']*transport)@initial
            out.append(state[0 if e['observable']=='temperature_amplitude' else 1])
        return np.array(out)
