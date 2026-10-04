from functools import lru_cache
import numpy as np
from scipy.linalg import eigh
from scipy.optimize import minimize_scalar

LENGTH=.05
AREA=1e-4
VOLUME=LENGTH*AREA
YOUNG=2e9
REFERENCE_TEMPERATURE=300.
HEAT_FIXED_STRAIN=1e6
BATH_CAPACITY=6.
MODES=64


def expansion(x):
    return .002*(1+.65*np.cos(2*np.pi*x/LENGTH))


@lru_cache(None)
def capacity_matrix():
    x=LENGTH*(np.arange(192)+.5)/192
    basis=np.sqrt(2)*np.cos(np.arange(MODES+1)[:,None]*np.pi*x[None,:]/LENGTH)
    basis[0]=1.
    alpha=expansion(x)
    local_heat=HEAT_FIXED_STRAIN+REFERENCE_TEMPERATURE*YOUNG*alpha**2
    mass=np.zeros((MODES+2,MODES+2))
    mass[:-1,:-1]=VOLUME*(basis*local_heat)@basis.T/len(x)
    mass[-1,-1]=BATH_CAPACITY
    return mass


@lru_cache(128)
def spectrum(conductivity,contact):
    if contact==0 and conductivity!=1.:
        rates,vectors,mass=spectrum(1.,0.)
        return conductivity*rates,vectors,mass
    mass=capacity_matrix()
    stiffness=np.diag(np.r_[conductivity*VOLUME*(np.arange(MODES+1)*np.pi/LENGTH)**2,0.])
    port=np.r_[1.,np.full(MODES,np.sqrt(2.)),-1.]
    stiffness+=contact*np.outer(port,port)
    rates,vectors=eigh(stiffness,mass)
    return np.maximum(rates,0),vectors,mass


class Model:
    def __init__(self):
        self.conductivity=None

    def fit(self,records):
        experiments=[r['input'] for r in records];values=np.array([r['value'] for r in records]);sigma=np.array([r['sigma'] for r in records])
        def loss(value):
            self.conductivity=value
            return np.sum(((self.predict(experiments)-values)/sigma)**2)
        answer=minimize_scalar(loss,bounds=(80.,220.),method='bounded',options={'xatol':1e-8})
        self.conductivity=float(answer.x)
        return self

    def predict(self,experiments):
        out=[]
        for e in experiments:
            initial=np.zeros(MODES+2)
            initial[:3]=[e['mean'],e['first']/np.sqrt(2),e['second']/np.sqrt(2)]
            initial[-1]=e['bath_initial']
            rates,vectors,mass=spectrum(self.conductivity,e['contact'])
            state=vectors@(np.exp(-rates*e['time'])*(vectors.T@(mass@initial)))
            values={'mean':state[0],'first':state[1]/np.sqrt(2),'second':state[2]/np.sqrt(2),'bath':state[-1]}
            out.append(values[e['observable']])
        return np.array(out)
