"""Velocity response above a driven sheet."""
import numpy as np
from scipy.optimize import minimize_scalar


def coefficients(viscosity,wave_number,frequency):
    k,w = wave_number,frequency
    s = np.sqrt(k*k-1j*w/viscosity)
    return np.array([k,s]),np.array([w*s/(k*(s-k)),-w/(s-k)])


def reading(e,viscosity):
    k,w = e['wave_number'],e['frequency']
    p,c = coefficients(viscosity,k,w)
    if e['observable']=='pumping':
        return float(-.5*np.real(np.sum(p*p*c)))
    field = c*np.exp(-p*e['height'])
    u = np.sum(-p*field)
    v = np.sum(-1j*k*field)
    return float({'u_real':u.real,'u_imag':u.imag,
                  'v_real':v.real,'v_imag':v.imag}[e['observable']])


def predict_at(experiments,viscosity):
    cache = {}
    out = []
    for e in experiments:
        key = tuple(sorted(e.items()))
        if key not in cache:
            cache[key] = reading(e,viscosity)
        out.append(cache[key])
    return np.array(out,dtype=float)


class Model:
    def __init__(self):
        self.viscosity = None

    def fit(self,records):
        inputs = [r['input'] for r in records]
        values = np.array([r['value'] for r in records])
        sigma = np.array([r['sigma'] for r in records])
        def objective(viscosity):
            residual = (predict_at(inputs,viscosity)-values)/sigma
            return float(residual@residual)
        fit = minimize_scalar(objective,bounds=(.7,1.4),method='bounded',
                              options={'xatol':1e-12})
        self.viscosity = float(min([fit.x,.7,1.4],key=objective))
        return self

    def predict(self,experiments):
        if self.viscosity is None:
            raise RuntimeError('Call fit before predict.')
        return predict_at(experiments,self.viscosity)
