"""Direct angular transport with shared temperature and displaced Bose drift."""
from functools import lru_cache
import numpy as np
from numpy.polynomial.legendre import leggauss,legval
from scipy.sparse.linalg import expm_multiply

PARAMETER='resistive_rate'
TRUE_PARAMETER=.3
SPEEDS=np.array([1.,.6])
CAPACITIES=np.array([3.,1./.6**3])


@lru_cache(4)
def angles(order):
    mu,w=leggauss(order);return mu,w/2


def generator(rate,normal,k,order):
    mu,w=angles(order);size=2*order
    energy=np.r_[CAPACITIES[0]*w,CAPACITIES[1]*w]/CAPACITIES.sum()
    momentum=np.r_[CAPACITIES[0]*w*mu/SPEEDS[0],CAPACITIES[1]*w*mu/SPEEDS[1]]
    common_temperature=np.ones(size)
    drift=np.r_[3*mu/SPEEDS[0],3*mu/SPEEDS[1]]/np.sum(CAPACITIES/SPEEDS**2)
    equilibrium=np.outer(common_temperature,energy)+np.outer(drift,momentum)
    resistive=np.zeros((size,size))
    for i in range(2):resistive[i*order:(i+1)*order,i*order:(i+1)*order]=np.outer(np.ones(order),w)
    return -1j*k*np.diag(np.r_[SPEEDS[0]*mu,SPEEDS[1]*mu])+rate*(resistive-np.eye(size))+normal*(equilibrium-np.eye(size))


def predict(experiments,resistive_rate=TRUE_PARAMETER,order=80):
    mu,w=angles(order);out=[]
    for e in experiments:
        initial=np.array([legval(mu,row) for row in e['initial']],dtype=complex).ravel()
        operator=generator(resistive_rate,e['normal_rate'],e['wavenumber'],order)
        final=expm_multiply(operator*e['time'],initial,traceA=np.trace(operator)*e['time']).reshape(2,order)
        polynomial=legval(mu,[0]*e['moment']+[1]);values=final@(w*polynomial)
        value=CAPACITIES@values/CAPACITIES.sum() if e['branch']==-1 else values[e['branch']]
        out.append(float(value.real if e['quadrature']=='cosine' else -value.imag))
    return np.array(out)


def reading(time,wavenumber=0.,normal_rate=.4,initial=((0.,0.,.08),(0.,0.,.1)),branch=0,moment=2,quadrature='cosine'):
    return dict(time=float(time),wavenumber=float(wavenumber),normal_rate=float(normal_rate),initial=[list(row) for row in initial],branch=int(branch),moment=int(moment),quadrature=quadrature)


def calibration_inputs():
    return [reading(t,normal_rate=n,branch=b) for n in [.2,.5,1.] for b in [0,1,-1] for t in np.geomspace(.05,3.,30)]


def hidden_inputs():
    reverse=-.12*(CAPACITIES[0]/SPEEDS[0])/(CAPACITIES[1]/SPEEDS[1])
    return {'counterflow':[reading(t,0.,2.,((0.,.12,0.),(0.,reverse,0.)),0,1) for t in np.linspace(.1,4.,24)],
            'energy_wave':[reading(t,1.2,2.5,((.06,0.,0.),(.06,0.,0.)),-1,0) for t in np.linspace(.3,7.,24)],
            'branch_flux':[reading(t,1.4,2.5,((.12,0.,0.),(0.,0.,0.)),1,1,'sine') for t in np.linspace(.2,6.,24)]}
