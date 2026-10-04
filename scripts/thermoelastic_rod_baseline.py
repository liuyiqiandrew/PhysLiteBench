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
MODES=96


def expansion(x):
    return .002*(1+.65*np.cos(2*np.pi*x/LENGTH))


def mechanical_viscosity(x):
    return YOUNG*12.*(1+.97*np.cos(2*np.pi*x/LENGTH))


@lru_cache(4)
def material_operators(modes=MODES):
    x=LENGTH*(np.arange(512)+.5)/512
    basis=np.sqrt(2)*np.cos(np.arange(modes+1)[:,None]*np.pi*x[None,:]/LENGTH)
    basis[0]=1.
    strain_basis=basis[1:]
    alpha=expansion(x)
    compliance=1/mechanical_viscosity(x)
    coupling=(strain_basis*alpha)@basis.T/len(x)
    mobility=(strain_basis*compliance)@strain_basis.T/len(x)

    # Availability is quadratic in temperature and the zero-mean strain modes.
    metric=np.diag(np.r_[np.full(modes+1,HEAT_FIXED_STRAIN*VOLUME/REFERENCE_TEMPERATURE),
                        np.full(modes,YOUNG*VOLUME),BATH_CAPACITY/REFERENCE_TEMPERATURE])
    difference=np.column_stack((-coupling,np.eye(modes),np.zeros(modes)))
    mechanical=YOUNG**2*VOLUME*difference.T@mobility@difference
    return metric,mechanical,coupling,mobility


@lru_cache(128)
def spectrum(conductivity,contact,modes=MODES):
    metric,mechanical,_,_=material_operators(modes)
    stiffness=np.zeros_like(metric)
    stiffness[:modes+1,:modes+1]=np.diag(conductivity*VOLUME*(np.arange(modes+1)*np.pi/LENGTH)**2)
    port=np.r_[1.,np.full(modes,np.sqrt(2.)),np.zeros(modes),-1.]
    stiffness+=contact*np.outer(port,port)
    dissipation=mechanical+stiffness/REFERENCE_TEMPERATURE
    rates,vectors=eigh(dissipation,metric,check_finite=False)
    return np.maximum(rates,0),vectors,metric


def initial_state(e,modes=MODES):
    initial=np.zeros(2*modes+2)
    initial[:3]=[e['mean'],e['first']/np.sqrt(2),e['second']/np.sqrt(2)]
    coupling=material_operators(modes)[2]
    initial[modes+1:-1]=coupling@initial[:modes+1]
    initial[-1]=e['bath_initial']
    return initial


class Model:
    def __init__(self):
        self.conductivity=None

    def fit(self,records):
        experiments=[r['input'] for r in records]
        values=np.array([r['value'] for r in records]);sigma=np.array([r['sigma'] for r in records])
        def loss(value):
            self.conductivity=value
            residual=(self.predict(experiments)-values)/sigma
            return residual@residual
        answer=minimize_scalar(loss,bounds=(80.,220.),method='bounded',options={'xatol':1e-8})
        self.conductivity=float(answer.x)
        return self

    def predict(self,experiments):
        out=[]
        for e in experiments:
            rates,vectors,metric=spectrum(self.conductivity,e['contact'])
            initial=initial_state(e)
            state=vectors@(np.exp(-rates*e['time'])*(vectors.T@(metric@initial)))
            values={'mean':state[0],'first':state[1]/np.sqrt(2),'second':state[2]/np.sqrt(2),'bath':state[-1]}
            out.append(values[e['observable']])
        return np.array(out)
