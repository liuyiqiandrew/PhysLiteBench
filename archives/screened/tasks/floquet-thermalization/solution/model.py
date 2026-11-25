from functools import lru_cache
import numpy as np
from scipy.special import jv
from scipy.optimize import minimize_scalar

TEMPERATURES=(.2,.65)
CUTOFFS=(1.5,.9)
STRENGTHS=(1.,.7)


def spectrum(frequency,bath):
    frequency=np.asarray(frequency,dtype=float)
    ratio=np.full_like(frequency,TEMPERATURES[bath])
    nonzero=np.abs(frequency)>1e-10
    ratio[nonzero]=frequency[nonzero]/(-np.expm1(-frequency[nonzero]/TEMPERATURES[bath]))
    return STRENGTHS[bath]*ratio*np.exp(-abs(frequency)/CUTOFFS[bath])


@lru_cache(512)
def coefficients(amplitude,frequency):
    harmonics=np.arange(-32,33)
    weights=jv(harmonics,amplitude/frequency)**2
    energies=1+harmonics*frequency
    down=np.array([weights@spectrum(energies,b) for b in [0,1]])
    up=np.array([weights@spectrum(-energies,b) for b in [0,1]])
    heat_down=np.array([(weights*energies)@spectrum(energies,b) for b in [0,1]])
    heat_up=np.array([(-weights*energies)@spectrum(-energies,b) for b in [0,1]])
    return down,up,heat_down,heat_up


def predict_at(experiments,coupling):
    out=[]
    for e in experiments:
        down,up,heat_down,heat_up=coefficients(e['amplitude'],e['frequency'])
        rate=coupling*np.sum(down+up)
        equilibrium=np.sum(up)/np.sum(down+up)
        time=2*np.pi*e['cycles']/e['frequency']
        initial=e['initial_population']
        if e['observable']=='population':
            value=equilibrium+(initial-equilibrium)*np.exp(-rate*time)
        else:
            mean_population=equilibrium+(initial-equilibrium)*(-np.expm1(-rate*time))/(rate*time)
            bath=e['bath']
            value=coupling*(heat_down[bath]*mean_population+heat_up[bath]*(1-mean_population))
        out.append(float(value))
    return np.array(out)


class Model:
    def __init__(self):
        self.coupling=None

    def fit(self,records):
        inputs=[r['input'] for r in records]
        values=np.array([r['value'] for r in records]);sigma=np.array([r['sigma'] for r in records])
        def objective(c):
            return np.sum(((predict_at(inputs,c)-values)/sigma)**2)
        self.coupling=float(minimize_scalar(objective,bounds=(.002,.008),method='bounded',options={'xatol':1e-13}).x)
        return self

    def predict(self,experiments):
        return predict_at(experiments,self.coupling)
