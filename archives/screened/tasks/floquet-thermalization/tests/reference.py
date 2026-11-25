"""FFT transition phase and bath-energy tilted population propagator."""
from functools import lru_cache
import numpy as np
from scipy.linalg import expm

PARAMETER='coupling'
TRUE_PARAMETER=.005


def experiment(amplitude=1.8,frequency=1.3,initial_population=.1,cycles=100,observable='heat',bath=0):
    return dict(amplitude=float(amplitude),frequency=float(frequency),initial_population=float(initial_population),cycles=int(cycles),observable=observable,bath=int(bath))


def calibration_inputs():
    return [experiment(0.,w,p,c,o,b) for w in [.8,1.3,1.9] for p in [.05,.9] for c in [10,40,100,180]
            for o,b in [('population',0),('heat',0),('heat',1)]]


def hidden_inputs():
    return {'cold_bath':[experiment(a,w,.1,c,'heat',0) for w in [.7,1.3,1.9] for a in [1.2,1.8,2.5] for c in [60,180]],
            'warm_bath':[experiment(a,w,.2,c,'heat',1) for w in [.7,1.3,1.9] for a in [1.2,1.8,2.5] for c in [60,180]],
            'frequency_scan':[experiment(2.1,w,.8,140,'heat',b) for w in np.linspace(.65,1.95,12) for b in [0,1]]}


def population_inputs():
    return [experiment(a,w,p,100,'population',0) for a in [1.,2.5] for w in [.65,1.3,1.95] for p in [.1,.8]]


@lru_cache(512)
def jumps(amplitude,frequency,samples=1024):
    theta=np.arange(samples)*2*np.pi/samples
    phase=np.exp(-1j*amplitude/frequency*np.sin(theta))
    coefficients=np.fft.fft(phase)/samples
    harmonics=np.fft.fftfreq(samples,1/samples).astype(int)
    keep=abs(coefficients)>1e-13
    energies=1-harmonics[keep]*frequency
    weights=abs(coefficients[keep])**2
    down=[];up=[]
    for temperature,cutoff,strength in [(.2,1.5,1.),(.65,.9,.7)]:
        def bath(w):
            x=abs(w)
            if x<1e-10:return strength*temperature
            occupation=0. if x/temperature>700 else 1/np.expm1(x/temperature)
            return strength*x*np.exp(-x/cutoff)*(occupation+(w>0))
        down.append(weights*np.array([bath(w) for w in energies]))
        up.append(weights*np.array([bath(-w) for w in energies]))
    return energies,np.array(down),np.array(up)


def generating_function(e,coupling,counting,samples=1024):
    energies,down,up=jumps(e['amplitude'],e['frequency'],samples)
    loss_down=coupling*np.sum(down);loss_up=coupling*np.sum(up)
    counted_down=down.astype(complex).copy();counted_up=up.astype(complex).copy()
    counted_down[e['bath']]*=np.exp(1j*counting*energies)
    counted_up[e['bath']]*=np.exp(-1j*counting*energies)
    generator=np.array([[-loss_down,coupling*np.sum(counted_up)],
                        [coupling*np.sum(counted_down),-loss_up]])
    time=2*np.pi*e['cycles']/e['frequency']
    return expm(generator*time)@np.array([e['initial_population'],1-e['initial_population']])


def predict(experiments,coupling=TRUE_PARAMETER,samples=1024,step=.0005):
    out=[]
    for e in experiments:
        if e['observable']=='population':
            value=generating_function(e,coupling,0.,samples)[0].real
        else:
            first=np.sum(generating_function(e,coupling,step,samples)).imag
            second=np.sum(generating_function(e,coupling,2*step,samples)).imag
            time=2*np.pi*e['cycles']/e['frequency']
            value=(8*first-second)/(6*step*time)
        out.append(float(value))
    return np.array(out)
