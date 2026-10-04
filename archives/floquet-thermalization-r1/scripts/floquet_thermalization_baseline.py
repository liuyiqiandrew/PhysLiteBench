import numpy as np
from scipy.optimize import minimize_scalar

TEMPERATURE=.2
CUTOFF=1.5


def spectrum(frequency):
    frequency=np.asarray(frequency,dtype=float)
    ratio=np.full_like(frequency,TEMPERATURE)
    nonzero=np.abs(frequency)>1e-10
    ratio[nonzero]=frequency[nonzero]/(-np.expm1(-frequency[nonzero]/TEMPERATURE))
    return ratio*np.exp(-abs(frequency)/CUTOFF)


def population(experiments,coupling):
    a=np.array([e['amplitude'] for e in experiments]);omega=np.array([e['frequency'] for e in experiments])
    cycles=np.array([e['cycles'] for e in experiments]);initial=np.array([e['initial_population'] for e in experiments])
    total_time=2*np.pi*cycles/omega
    if np.all(a==0):
        up=coupling*float(spectrum(-1.));down=coupling*float(spectrum(1.));equilibrium=up/(up+down)
        return equilibrium+(initial-equilibrium)*np.exp(-(up+down)*total_time)
    # Compose the positive instantaneous thermal rate equation through one
    # period, then repeat its affine propagator for the requested cycles.
    steps=512;dt=2*np.pi/(omega*steps);slope=np.ones(len(a));offset=np.zeros(len(a))
    for phase in (np.arange(steps)+.5)*2*np.pi/steps:
        gap=1+a*np.cos(phase)
        up=coupling*spectrum(-gap);down=coupling*spectrum(gap);rate=up+down
        decay=np.exp(-rate*dt)
        slope*=decay;offset=offset*decay+up/rate*(-np.expm1(-rate*dt))
    periodic=offset/(-np.expm1(np.log(slope)))
    return periodic+(initial-periodic)*slope**cycles



class Model:
    def __init__(self):
        self.coupling=None

    def fit(self,records):
        inputs=[r['input'] for r in records]
        y=np.array([r['value'] for r in records]);sigma=np.array([r['sigma'] for r in records])
        self.coupling=float(minimize_scalar(lambda c: np.sum(((population(inputs,c)-y)/sigma)**2),bounds=(.002,.008),method='bounded',options={'xatol':1e-13}).x)
        return self

    def predict(self,experiments):
        return population(experiments,self.coupling)
