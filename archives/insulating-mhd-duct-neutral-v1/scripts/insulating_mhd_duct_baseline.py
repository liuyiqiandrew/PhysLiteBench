import numpy as np
from scipy.optimize import minimize_scalar

WIDTH, HEIGHT = .012, .008
DENSITY, CONDUCTIVITY = 1000., 1e5
X_MODES, Y_MODES = 25, 35
M = np.arange(1, 2*X_MODES, 2, dtype=float)
N = np.arange(1, 2*Y_MODES, 2, dtype=float)
MEAN_X, MEAN_Y = 2*np.sqrt(2)/(np.pi*M), 2*np.sqrt(2)/(np.pi*N)


class Model:
    def __init__(self):
        self.viscosity = None

    def fit(self, records):
        experiments=[r['input'] for r in records]
        observed=np.array([r['value'] for r in records]);sigma=np.array([r['sigma'] for r in records])
        def loss(value):
            self.viscosity=value
            return np.sum(((self.predict(experiments)-observed)/sigma)**2)
        answer=minimize_scalar(loss,bounds=(.006,.018),method='bounded',options={'xatol':1e-12})
        self.viscosity=float(answer.x)
        return self

    def predict(self, experiments):
        out=[]
        for e in experiments:
            rates=self.viscosity/DENSITY*((M[:,None]*np.pi/WIDTH)**2+(N[None,:]*np.pi/HEIGHT)**2)+CONDUCTIVITY*e['magnetic_field']**2/DENSITY
            amplitudes=-np.expm1(-rates*e['time'])/rates*MEAN_X[:,None]*MEAN_Y[None,:]
            if e['observable']=='mean':
                weight=MEAN_X[:,None]*MEAN_Y[None,:]
            else:
                weight=2*np.sin(M[:,None]*np.pi*e['x'])*np.sin(N[None,:]*np.pi*e['y'])
            out.append(e['pressure_gradient']/DENSITY*np.sum(weight*amplitudes))
        return np.array(out)
