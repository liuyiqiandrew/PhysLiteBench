"""Independent fixed-number hard-core boson evolution."""
from itertools import combinations
from functools import lru_cache
import numpy as np
from scipy.linalg import eigh

SITES=8
OCCUPIED=(2,3,4)
POSITIONS=np.arange(SITES)-3.5
TRUE_PARAMETER=1.

@lru_cache(maxsize=2048)
def exact_fock(hopping,trap,tilt,duration):
    basis=[sum(1<<j for j in sites) for sites in combinations(range(SITES),len(OCCUPIED))]
    index={b:i for i,b in enumerate(basis)}
    H=np.zeros((len(basis),len(basis)))
    onsite=trap*POSITIONS**2+tilt*POSITIONS
    for col,bits in enumerate(basis):
        H[col,col]=sum(onsite[j] for j in range(SITES) if bits&(1<<j))
        for j in range(SITES-1):
            if bool(bits&(1<<j)) != bool(bits&(1<<(j+1))):
                target=bits^(1<<j)^(1<<(j+1));H[index[target],col]=-hopping
    energies,U=eigh(H)
    initial=index[sum(1<<j for j in OCCUPIED)]
    psi=(U*np.exp(-1j*energies*duration))@U[initial].conj()
    rho=np.zeros((SITES,SITES),complex)
    for j in range(SITES):
        for i in range(SITES):
            for col,bits in enumerate(basis):
                if not bits&(1<<j):continue
                removed=bits^(1<<j)
                if removed&(1<<i):continue
                target=removed|(1<<i)
                rho[i,j]+=np.conj(psi[index[target]])*psi[col]
    return rho,psi,H


def momentum(rho,q):
    amplitude=np.exp(-1j*q*POSITIONS)
    return float(np.vdot(amplitude,rho@amplitude).real/SITES)



def predict(experiments,hopping=TRUE_PARAMETER):
    result=[]
    for e in experiments:
        rho,_,_=exact_fock(hopping,e['trap'],e['tilt'],e['duration'])
        result.append(float(rho[e['site'],e['site']].real) if e['observable']=='density' else momentum(rho,e['wave_number']))
    return np.asarray(result)


def experiment(trap,tilt,duration,observable='momentum',detector=0.):
    e=dict(trap=trap,tilt=tilt,duration=duration,observable=observable)
    e['site' if observable=='density' else 'wave_number']=detector
    return e


def calibration_inputs():
    return [experiment(a,b,t,'density',site) for a,b in [(0.,0.),(.08,.15),(.15,-.3)] for t in [.12,.2,.3,.4] for site in [1,5]]*12


def hidden_inputs():
    qs=np.linspace(-np.pi,np.pi,17,endpoint=False)
    groups={label:[experiment(a,b,t,'momentum',float(q)) for t in [.6,1.,1.5] for q in qs] for label,a,b in [('free_release',0.,0.),('trapped_release',.08,.15),('tilted_release',.15,-.3)]}
    groups['density_anchors']=[experiment(a,b,t,'density',j) for a,b in [(0.,0.),(.08,.15),(.15,-.3)] for t in [.5,1.2,2.] for j in range(8)]
    groups['initial_momentum']=[experiment(.12,-.2,0.,'momentum',float(q)) for q in qs]
    return groups
