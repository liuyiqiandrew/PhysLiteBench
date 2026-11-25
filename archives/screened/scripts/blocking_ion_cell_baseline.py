from functools import lru_cache
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize_scalar

LENGTH=1e-6
PERMITTIVITY=7e-10
CONCENTRATION=.015
FARADAY=96485.33212
THERMAL_VOLTAGE=8.314462618*300/FARADAY
COUPLING=FARADAY*CONCENTRATION*LENGTH**2/(PERMITTIVITY*THERMAL_VOLTAGE)
CELLS=96


def bernoulli(value):
    out=np.ones_like(value)
    np.divide(value,np.expm1(value),out=out,where=abs(value)>1e-8)
    small=abs(value)<=1e-8
    out[small]=1-value[small]/2+value[small]**2/12
    return out


def electric_state(concentration,voltage,capacitance):
    cells=concentration.shape[1];dx=1./cells;x=(np.arange(cells)+.5)*dx
    density=concentration[0]-concentration[1]
    polarization=dx*((1-x)@density)
    ratio=PERMITTIVITY/(capacitance*LENGTH)
    charge=(voltage/THERMAL_VOLTAGE/COUPLING-polarization)/(1+ratio)
    field=np.full(cells-1,COUPLING*(charge+polarization))
    return charge,field


@lru_cache(32)
def trajectory(first,second,voltage,capacitance):
    dx=1./CELLS;x=(np.arange(CELLS)+.5)*dx
    initial=1+first*np.cos(np.pi*x)+second*np.cos(2*np.pi*x)
    def rhs(time,state):
        c=state.reshape(2,CELLS)
        charge,field=electric_state(c,voltage,capacitance)
        step=np.array([1.,-1.])[:,None]*field[None,:]*dx
        flux=(bernoulli(-step)*c[:,:-1]-bernoulli(step)*c[:,1:])/dx
        derivative=np.zeros_like(c)
        derivative[:,:-1]-=flux/dx;derivative[:,1:]+=flux/dx
        return derivative.ravel()
    answer=solve_ivp(rhs,(0.,16.),np.tile(initial,2),method='BDF',rtol=2e-9,atol=2e-11,dense_output=True)
    if not answer.success:raise RuntimeError(answer.message)
    return answer.sol


class Model:
    def __init__(self):
        self.diffusivity=None

    def fit(self,records):
        experiments=[r['input'] for r in records];values=np.array([r['value'] for r in records]);sigma=np.array([r['sigma'] for r in records])
        def loss(value):
            self.diffusivity=value*1e-9
            return np.sum(((self.predict(experiments)-values)/sigma)**2)
        answer=minimize_scalar(loss,bounds=(.6,1.6),method='bounded',options={'xatol':1e-11})
        self.diffusivity=float(answer.x)*1e-9
        return self

    def predict(self,experiments):
        out=[];x=(np.arange(CELLS)+.5)/CELLS
        for e in experiments:
            if e['voltage']==0:
                if e['observable']=='electrode_charge':out.append(0.);continue
                mode=1 if e['observable'].endswith('first') else 2
                amplitude=e['first'] if mode==1 else e['second']
                out.append(amplitude/2*np.exp(-self.diffusivity*(mode*np.pi/LENGTH)**2*e['time']));continue
            solution=trajectory(e['first'],e['second'],e['voltage'],e['series_capacitance'])
            c=solution(self.diffusivity*e['time']/LENGTH**2).reshape(2,CELLS)
            if e['observable']=='electrode_charge':out.append(electric_state(c,e['voltage'],e['series_capacitance'])[0])
            else:
                species=0 if e['observable'].startswith('positive') else 1
                mode=1 if e['observable'].endswith('first') else 2
                out.append(c[species]@np.cos(mode*np.pi*x)/CELLS)
        return np.array(out)
