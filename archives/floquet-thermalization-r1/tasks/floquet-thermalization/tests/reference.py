"""Fourier analysis of the exact driven transition phase."""
import numpy as np
from functools import lru_cache

PARAMETER='coupling'
TRUE_PARAMETER=.005


def experiment(amplitude=1.8,frequency=1.3,initial_population=.1,cycles=80):
    return dict(amplitude=float(amplitude),frequency=float(frequency),initial_population=float(initial_population),cycles=int(cycles))


def calibration_inputs():
    return [experiment(0.,w,p,c) for w in [.8,1.3,1.9] for p in [.05,.9] for c in np.linspace(8,180,20,dtype=int)]


def hidden_inputs():
    return {'rapid_modulation':[experiment(a,w,.05,c) for w in [1.3,1.7,1.9] for a in [1.5,2.,2.5] for c in [60,120,190]],
            'population_decay':[experiment(a,w,.8,c) for w in [.7,1.1,1.6] for a in [1.5,2.,2.5] for c in [40,100,180]],
            'variable_frequency':[experiment(2.1,w,.2,c) for w in np.linspace(.65,1.95,10) for c in [60,140,200]]}


@lru_cache(512)
def rates(amplitude,frequency,samples=2048):
    theta=np.arange(samples)*2*np.pi/samples
    # Strip the carrier exp(i*t) from U_g^*(t) U_e(t). Its remaining periodic
    # phase is obtained by integrating the stated sinusoidal level splitting.
    periodic=np.exp(-1j*amplitude/frequency*np.sin(theta))
    coefficients=np.fft.fft(periodic)/samples
    integers=np.fft.fftfreq(samples,1/samples).astype(int)
    energies=1-integers*frequency
    def bath(w):
        x=abs(w);occupation=0. if x/.2>700 else 1/np.expm1(x/.2) if x>1e-10 else None
        if x<1e-10:return .2
        return x*np.exp(-x/1.5)*(occupation+(w>0))
    down=sum(abs(z)**2*bath(w) for z,w in zip(coefficients,energies) if abs(z)>1e-14)
    up=sum(abs(z)**2*bath(-w) for z,w in zip(coefficients,energies) if abs(z)>1e-14)
    return float(up),float(down)


def predict(experiments,coupling,samples=2048):
    out=[]
    for e in experiments:
        up,down=np.array(rates(e['amplitude'],e['frequency'],samples))*coupling
        time=e['cycles']*2*np.pi/e['frequency'];p_eq=up/(up+down)
        out.append(float(p_eq+(e['initial_population']-p_eq)*np.exp(-(up+down)*time)))
    return np.array(out)
