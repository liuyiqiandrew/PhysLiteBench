import numpy as np
from scipy.optimize import minimize_scalar

POISSON=.48
VISCOSITY=.15


def compliance(wavenumber,thickness,young_modulus):
    k=float(wavenumber)*1000
    depth=float(thickness)*.001
    modulus=float(young_modulus)*1e6
    longitudinal=modulus*(1-POISSON)/((1+POISSON)*(1-2*POISSON))
    return depth/longitudinal


def predict_at(experiments,young_modulus):
    out=[]
    for e in experiments:
        c=compliance(e['wavenumber'],e['thickness'],young_modulus)
        k=e['wavenumber']*1000
        gap=e['gap']*1e-6
        decay=gap**3*k**2/(12*VISCOSITY*c)
        out.append(1e9*e['pressure']*c*np.exp(-decay*e['elapsed_time']))
    return np.array(out)


class Model:
    def __init__(self):
        self.young_modulus=None

    def fit(self,records):
        inputs=[r['input'] for r in records]
        values=np.array([r['value'] for r in records]);sigma=np.array([r['sigma'] for r in records])
        def objective(modulus):
            return np.sum(((predict_at(inputs,modulus)-values)/sigma)**2)
        self.young_modulus=float(minimize_scalar(objective,bounds=(.7,1.6),method='bounded',options={'xatol':1e-13}).x)
        return self

    def predict(self,experiments):
        return predict_at(experiments,self.young_modulus)
