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
        times = np.array([r['input']['time'] for r in records])
        rates = (np.pi*np.array([r['input']['mode'] for r in records])/np.array([r['input']['length'] for r in records]))**2
        amplitude = np.array([r['input']['composition_amplitude'] for r in records])
        values = np.array([r['value'] for r in records])
        sigma = np.array([r['sigma'] for r in records])
        def loss(d):
            return np.sum(((amplitude*np.exp(-d*rates*times)-values)/sigma)**2)
        result = minimize_scalar(loss,bounds=(.01,.08),method='bounded',options={'xatol':1e-12})
        self.diffusivity = float(result.x)
        return self

    def predict(self, experiments):
        out = []
        d = self.diffusivity
        transport = np.array([[CONDUCTIVITY*1e6/CAPACITY,DENSITY*d*GAS_CONSTANT*T0*T0*SORET/CAPACITY],
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
