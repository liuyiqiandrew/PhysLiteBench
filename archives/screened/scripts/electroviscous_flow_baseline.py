from functools import lru_cache
import numpy as np
from scipy.integrate import solve_bvp
from numpy.polynomial.legendre import leggauss

PERMITTIVITY=7e-10
TEMPERATURE=300.
GAS_CONSTANT=8.314462618
FARADAY=96485.33212
DIFFUSIVITY=1e-10
THERMAL_VOLTAGE=GAS_CONSTANT*TEMPERATURE/FARADAY
NODES,WEIGHTS=leggauss(128)
NODES=(NODES+1)/2
WEIGHTS=WEIGHTS/2


@lru_cache(256)
def equilibrium(half_gap,concentration,surface_charge):
    h=half_gap*1e-6
    squared_screening=2*FARADAY**2*concentration*h*h/(PERMITTIVITY*GAS_CONSTANT*TEMPERATURE)
    wall_derivative=h*surface_charge/(PERMITTIVITY*THERMAL_VOLTAGE)
    x=np.linspace(0,1,80)
    guess=np.arcsinh(wall_derivative/squared_screening)
    solution=solve_bvp(lambda x,y:np.vstack([y[1],squared_screening*np.sinh(y[0])]),
        lambda a,b:np.array([a[1],b[1]-wall_derivative]),x,np.vstack([np.full_like(x,guess),np.zeros_like(x)]),tol=1e-9,max_nodes=5000)
    if not solution.success:
        raise RuntimeError(solution.message)
    return solution.sol


class Model:
    def __init__(self):
        self.viscosity=None

    def fit(self,records):
        coefficients=[]
        for r in records:
            e=r['input'];h=e['half_gap']*1e-6
            shape=1/3 if e['observable']=='mean' else (1-e['y']**2)/2
            coefficients.append(e['pressure_gradient']*h*h*shape)
        coefficient=np.array(coefficients)
        values=np.array([r['value'] for r in records]);sigma=np.array([r['sigma'] for r in records])
        self.viscosity=float(np.clip(np.sum((coefficient/sigma)**2)/np.sum(coefficient*values/sigma**2),.0008,.002))
        return self

    def predict(self,experiments):
        out=[]
        for e in experiments:
            h=e['half_gap']*1e-6;g=e['pressure_gradient'];c=e['concentration']
            potential=equilibrium(e['half_gap'],c,e['surface_charge'])
            psi=potential(NODES)[0];wall=potential(1.)[0]
            charge=-2*FARADAY*c*np.sinh(psi)
            conductivity=2*FARADAY**2*c*DIFFUSIVITY/(GAS_CONSTANT*TEMPERATURE)*np.cosh(psi)
            pressure=g*h*h*(1-NODES*NODES)/(2*self.viscosity)
            electroosmotic=PERMITTIVITY*THERMAL_VOLTAGE*(psi-wall)/self.viscosity
            streaming=WEIGHTS@(charge*pressure)
            conductance=WEIGHTS@conductivity
            field=-streaming/conductance
            if e['observable']=='mean':
                out.append(WEIGHTS@(pressure+electroosmotic*field))
            else:
                y=e['y'];psi_y=potential(y)[0]
                out.append(g*h*h*(1-y*y)/(2*self.viscosity)+PERMITTIVITY*THERMAL_VOLTAGE*(psi_y-wall)*field/self.viscosity)
        return np.array(out)
