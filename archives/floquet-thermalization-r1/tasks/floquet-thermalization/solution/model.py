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
    from scipy.special import jv
    out=[];harmonics=np.arange(-32,33)
    for e in experiments:
        weights=jv(harmonics,e['amplitude']/e['frequency'])**2
        energies=1+harmonics*e['frequency']
        up=coupling*np.sum(weights*spectrum(-energies));down=coupling*np.sum(weights*spectrum(energies))
        equilibrium=up/(up+down);time=2*np.pi*e['cycles']/e['frequency']
        out.append(equilibrium+(e['initial_population']-equilibrium)*np.exp(-(up+down)*time))
    return np.array(out)



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
