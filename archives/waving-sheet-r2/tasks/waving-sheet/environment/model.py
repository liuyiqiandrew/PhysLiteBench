"""Velocity response above a driven sheet in a polymer solution."""
import numpy as np


def coefficients(viscosity,wave_number,frequency):
    k,w = wave_number,frequency
    viscosity_star = viscosity*(.25+.75/(1-1j*w))
    s = np.sqrt(k*k-1j*w/viscosity_star)
    return np.array([k,s]),np.array([w*s/(k*(s-k)),-w/(s-k)])


def reading(e,viscosity):
    k,w = e['wave_number'],e['frequency']
    p,c = coefficients(viscosity,k,w)
    if e['observable']=='pumping':
        boundary = -.5*np.real(np.sum(p*p*c))
        u,v = -p*c,-1j*k*c
        flux = .5*np.real(np.sum(u[:,None]*v.conj()[None,:]/
                    (p[:,None]+p.conj()[None,:])))
        return float(boundary+flux/viscosity)
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
        raise NotImplementedError('Fit viscosity from the calibration records.')

    def predict(self,experiments):
        if self.viscosity is None:
            raise RuntimeError('Call fit before predict.')
        return predict_at(experiments,self.viscosity)
