"""Independent finite-volume entropy/strain evolution with algebraic force balance."""
from functools import lru_cache
import numpy as np
from scipy.linalg import eig, solve, helmert

PARAMETER='conductivity'
TRUE_PARAMETER=145.
BOUNDS=(80.,220.)


@lru_cache(32)
def entropy_operator(conductivity,contact,cells):
    length,area,t0,young,heat,bath=.05,1e-4,300.,2e9,1e6,6.
    dx=length/cells;volume=area*dx;alpha_scale=.002
    x=(np.arange(cells)+.5)/cells
    alpha=.002*(1+.65*np.cos(2*np.pi*x))
    drag=young*12*(1+.97*np.cos(2*np.pi*x))
    # State: T0*entropy_increment/c_e, strain/alpha_scale, bath temperature.
    strain_basis=helmert(cells,full=False).T
    temperature=np.zeros((cells+1,2*cells))
    temperature[:cells,:cells]=np.eye(cells)
    temperature[:cells,cells:-1]=-np.diag(t0*young/heat*alpha*alpha_scale)@strain_basis
    temperature[-1,-1]=1.
    unconstrained=young*alpha[:,None]*temperature[:cells]
    unconstrained[:,cells:-1]-=young*alpha_scale*strain_basis
    # Total stress is uniform, and its value makes the mean strain rate zero.
    reaction=-(1/drag)@unconstrained/np.sum(1/drag)
    strain_rate=(unconstrained+reaction[None,:])/drag[:,None]/alpha_scale
    stiffness=np.zeros((cells+1,cells+1))
    for i in range(cells-1):
        v=np.zeros(cells+1);v[i],v[i+1]=1.,-1.
        stiffness+=conductivity*area/dx*np.outer(v,v)
    effective=0. if contact==0 else 1/(1/contact+dx/(2*conductivity*area))
    v=np.zeros(cells+1);v[0],v[-1]=1.,-1.
    stiffness+=effective*np.outer(v,v)
    heat_rate=-stiffness@temperature
    generator=np.vstack((heat_rate[:cells]/(heat*volume),strain_basis.T@strain_rate,heat_rate[-1:]/bath))
    rates,vectors=eig(generator)
    assert max(abs(rates.imag))<1e-7
    return rates.real,temperature@vectors.real,solve(vectors.real,np.eye(2*cells))


def predict(experiments,conductivity,cells=128):
    out=[];x=(np.arange(cells)+.5)/cells
    alpha=.002*(1+.65*np.cos(2*np.pi*x))
    for e in experiments:
        initial_t=e['mean']+e['first']*np.cos(np.pi*x)+e['second']*np.cos(2*np.pi*x)
        strain=alpha*initial_t-np.mean(alpha*initial_t)
        entropy=initial_t+300.*2e9/1e6*alpha*strain
        initial=np.r_[entropy,helmert(cells,full=False)@(strain/.002),e['bath_initial']]
        rates,temperature_modes,inverse_modes=entropy_operator(conductivity,e['contact'],cells)
        state=temperature_modes@(np.exp(rates*e['time'])*(inverse_modes@initial))
        values={'mean':state[:-1].mean(),'bath':state[-1],
                'first':state[:-1]@np.cos(np.pi*x)/cells,
                'second':state[:-1]@np.cos(2*np.pi*x)/cells}
        out.append(values[e['observable']])
    return np.array(out)


def calibration_inputs():
    return [dict(mean=0.,first=.5,second=0.,bath_initial=0.,contact=0.,
                 time=float(t),observable='first') for t in np.geomspace(.15,40.,80)]


def hidden_inputs():
    result={f'contact_{h}':[dict(mean=a,first=0.,second=c,bath_initial=bath,
                                contact=h,time=float(t),observable=name)
            for name in ['mean','second','bath'] for t in np.geomspace(.4,80.,24)]
            for h,a,c,bath in [(.4,0.,.9,-.9),(1.1,.35,.6,-.9)]}
    result['insulated_even']=[dict(mean=0.,first=0.,second=.9,bath_initial=0.,
                                   contact=0.,time=float(t),observable=name)
            for name in ['mean','second'] for t in np.geomspace(.4,80.,24)]
    return result
