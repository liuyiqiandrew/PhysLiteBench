"""Independent occupation-space Hamiltonian and detector transition rates."""
from functools import lru_cache
import numpy as np
from scipy.linalg import eigh

TRUE_PARAMETER=1.1


def annihilator(orbital):
    result=np.zeros((64,64),complex)
    for state in range(64):
        if state&(1<<orbital):
            sign=(-1)**((state&((1<<orbital)-1)).bit_count())
            result[state^(1<<orbital),state]=sign
    return result


C=[annihilator(a) for a in range(6)]
DENSITY=[c.conj().T@c for c in C]


@lru_cache(maxsize=512)
def modes(hopping, pairing, offset, phase):
    H=np.zeros((64,64),complex)
    for i,onsite in enumerate([-.35,.15,.7]):
        H+=(onsite+offset)*(DENSITY[2*i]+DENSITY[2*i+1])
    for i,j,w in [(0,1,1.),(1,2,.8),(0,2,.3)]:
        for spin in [0,1]:
            term=-hopping*w*C[2*i+spin].conj().T@C[2*j+spin]
            H+=term+term.conj().T
    for i,(d,b) in enumerate(zip([1.,.85,1.15],[0.,1.,-.4])):
        term=pairing*d*np.exp(1j*phase*b)*C[2*i].conj().T@C[2*i+1].conj().T
        H+=term+term.conj().T
    return eigh(H)


def rate(e):
    energy,V=modes(e['hopping'],e['pairing'],e['offset'],e['phase'])
    population=np.exp(-(energy-energy[0])/e['temperature'])
    population/=population.sum()
    site=e['site']
    operator=(DENSITY[2*site]-DENSITY[2*site+1])/2
    matrix=V.conj().T@operator@V
    gap=energy[:,None]-energy[None,:]
    window=np.where(gap>0.,gap**2*np.exp(-.5*((gap-e['center'])/e['width'])**2),0.)
    return float(np.sum(population[None,:]*abs(matrix)**2*window))


def predict(experiments,gain=TRUE_PARAMETER):
    return gain*np.array([rate(e) for e in experiments])


def experiment(hopping=.8,pairing=.7,temperature=.55,site=0,center=2.6,width=.65,phase=0.,offset=0.):
    return dict(hopping=hopping,pairing=pairing,temperature=temperature,site=site,center=center,width=width,phase=phase,offset=offset)


def calibration_inputs():
    return [experiment(hopping=h,pairing=0.,site=i,temperature=t,center=c,width=.7)
            for _ in range(4) for h in [.55,.8,1.] for i in [0,1,2]
            for t in [.3,.6] for c in [1.7,2.5]]


def hidden_inputs():
    return {
        'pairing_sweep':[experiment(pairing=g,site=0) for g in [.45,.65,.85,1.05]],
        'phase_sweep':[experiment(hopping=.85,pairing=.8,site=1,center=2.5,phase=a,temperature=.5) for a in [-1.2,-.5,.5,1.2]],
        'detector_sweep':[experiment(hopping=.85,pairing=.65,site=2,center=c,temperature=.6) for c in [1.8,2.1,2.4,2.7]],
        'normal_anchors':[experiment(hopping=.65,pairing=0.,site=0,temperature=.4,center=1.8),experiment(hopping=.95,pairing=0.,site=2,temperature=.7,center=2.3)]}
