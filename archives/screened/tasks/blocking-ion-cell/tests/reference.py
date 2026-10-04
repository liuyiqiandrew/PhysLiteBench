"""Independent SI-unit Poisson/circuit solve and central conservative ion fluxes."""
from functools import lru_cache
import numpy as np
from scipy.integrate import solve_ivp
from scipy.sparse import diags,lil_matrix
from scipy.sparse.linalg import splu

PARAMETER='diffusivity'
TRUE_PARAMETER=1.1e-9
BOUNDS=(.6e-9,1.6e-9)


@lru_cache(16)
def potential_solver(capacitance,cells):
    dx=1e-6/cells
    matrix=lil_matrix((cells+1,cells+1))
    matrix[:cells,:cells]=diags([-np.ones(cells-1),np.r_[3.,np.full(cells-2,2.),3.],-np.ones(cells-1)],[-1,0,1])
    matrix[0,-1]=-2.
    matrix[-1,0]=-2*7e-10/(capacitance*dx)
    matrix[-1,-1]=1+2*7e-10/(capacitance*dx)
    return splu(matrix.tocsc())


def fields(c,voltage,capacitance):
    cells=c.shape[1];dx=1e-6/cells
    rhs=np.r_[96485.33212/7e-10*(c[0]-c[1])*dx**2,voltage]
    potential=potential_solver(capacitance,cells).solve(rhs)
    field=-np.diff(potential[:-1])/dx
    charge=7e-10*2*(potential[-1]-potential[0])/dx
    return field,charge,potential


@lru_cache(16)
def trajectory(first,second,voltage,capacitance,diffusivity,cells):
    dx=1e-6/cells;x=(np.arange(cells)+.5)/cells
    initial=.015*(1+first*np.cos(np.pi*x)+second*np.cos(2*np.pi*x))
    def rhs(time,state):
        c=state.reshape(2,cells)
        field,charge,potential=fields(c,voltage,capacitance)
        drift=np.array([1.,-1.])[:,None]*96485.33212/(8.314462618*300)*field[None,:]
        flux=diffusivity*(-np.diff(c,axis=1)/dx+drift*(c[:,:-1]+c[:,1:])/2)
        derivative=np.zeros_like(c)
        derivative[:,:-1]-=flux/dx;derivative[:,1:]+=flux/dx
        return derivative.ravel()
    answer=solve_ivp(rhs,(0.,.01),np.tile(initial,2),method='BDF',rtol=2e-9,atol=2e-13,dense_output=True)
    assert answer.success,answer.message
    return answer.sol


def predict(experiments,diffusivity,cells=160):
    out=[];x=(np.arange(cells)+.5)/cells
    for e in experiments:
        if e['voltage']==0:
            if e['observable']=='electrode_charge':out.append(0.);continue
            mode=1 if e['observable'].endswith('first') else 2
            amplitude=e['first'] if mode==1 else e['second']
            out.append(amplitude/2*np.exp(-diffusivity*(mode*np.pi/1e-6)**2*e['time']));continue
        state=trajectory(e['first'],e['second'],e['voltage'],e['series_capacitance'],diffusivity,cells)(e['time']).reshape(2,cells)
        if e['observable']=='electrode_charge':out.append(fields(state,e['voltage'],e['series_capacitance'])[1]/(96485.33212*.015*1e-6))
        else:
            species=0 if e['observable'].startswith('positive') else 1
            mode=1 if e['observable'].endswith('first') else 2
            out.append(state[species]@np.cos(mode*np.pi*x)/(.015*cells))
    return np.array(out)


def calibration_inputs():
    return [dict(first=.5,second=-.25,voltage=0.,series_capacitance=.003,time=float(t),observable=obs)
            for obs in ['positive_first','negative_second'] for t in np.geomspace(1e-6,.001,40)]


def hidden_inputs():
    return {name:[dict(first=a,second=b,voltage=v,series_capacitance=c,time=float(t),observable=obs)
            for obs in ['positive_first','negative_first','positive_second','electrode_charge']
            for t in np.geomspace(1e-5,.004,18)]
            for name,a,b,v,c in [('positive_step',0.,0.,.065,.003),('negative_step',.3,-.15,-.08,.0015),('larger_capacitor',-.2,.1,.08,.007)]}
