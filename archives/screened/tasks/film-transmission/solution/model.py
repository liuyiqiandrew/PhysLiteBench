import numpy as np
from scipy.optimize import minimize_scalar

FILM_INDEX = 2.1


def amplitudes(experiments,thickness):
    wavelength=np.array([e['wavelength'] for e in experiments])
    theta=np.array([e['angle'] for e in experiments])
    exit_index=np.array([e['exit_index'] for e in experiments])
    parallel=np.sin(theta)
    left=np.cos(theta)
    film=np.sqrt(FILM_INDEX**2-parallel**2)
    right=np.sqrt(exit_index**2-parallel**2)
    phase=2*np.pi*thickness*film/wavelength
    c,s=np.cos(phase),np.sin(phase)
    m00,m01,m10,m11=c,-1j*s/film,-1j*film*s,c
    denominator=left*m00+left*right*m01+m10+right*m11
    reflected=(left*m00+left*right*m01-m10-right*m11)/denominator
    transmitted=2*left/denominator
    return reflected,transmitted


class Model:
    def __init__(self):
        self.thickness=None

    def fit(self,records):
        experiments=[r['input'] for r in records]
        values=np.array([r['value'] for r in records])
        sigma=np.array([r['sigma'] for r in records])
        def loss(thickness):
            self.thickness=thickness
            return np.sum(((self.predict(experiments)-values)/sigma)**2)
        fit=minimize_scalar(loss,bounds=(.05,.12),method='bounded',options={'xatol':1e-13})
        self.thickness=float(fit.x)
        return self

    def predict(self,experiments):
        reflected,transmitted=amplitudes(experiments,self.thickness)
        angle=np.array([e['angle'] for e in experiments])
        index=np.array([e['exit_index'] for e in experiments])
        admittance_ratio=np.sqrt(index**2-np.sin(angle)**2)/np.cos(angle)
        reflection=np.abs(reflected)**2
        transmission=admittance_ratio*np.abs(transmitted)**2
        return np.where([e['observable']=='reflectance' for e in experiments],reflection,transmission)
