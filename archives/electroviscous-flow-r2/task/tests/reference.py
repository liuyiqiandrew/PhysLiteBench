"""Independent joint collocation of electrostatics, momentum, and total current."""
from functools import lru_cache
import numpy as np
from scipy.integrate import solve_bvp
from numpy.polynomial.legendre import leggauss

PARAMETER='viscosity'
TRUE_PARAMETER=.0012
BOUNDS=(.0008,.002)


@lru_cache(256)
def solve_channel(half_gap,concentration,surface_charge,viscosity,tolerance=2e-9):
    h=half_gap*1e-6;eps=7e-10;R=8.314462618;T=300.;F=96485.33212;D=1e-10
    voltage=R*T/F
    screening=2*F*F*concentration*h*h/(eps*R*T)
    wall=h*surface_charge/(eps*voltage)
    coupling=2*concentration*R*T*h*h/(viscosity*D)
    def rhs(x,y,p):
        psi,dpsi,v,dv,current=y
        return np.vstack([dpsi,screening*np.sinh(psi),dv,
                           -np.ones_like(x)+coupling*np.sinh(psi)*p[0],
                           np.cosh(psi)*p[0]-np.sinh(psi)*v])
    def boundary(a,b,p):
        return np.array([a[1],b[1]-wall,a[3],b[2],a[4],b[4]])
    x=np.linspace(0,1,100);guess=np.arcsinh(wall/screening)
    y=np.vstack([np.full_like(x,guess),wall*x,(1-x*x)/2,-x,np.zeros_like(x)])
    sol=solve_bvp(rhs,boundary,x,y,p=[0.],tol=tolerance,max_nodes=10000)
    if not sol.success:raise RuntimeError(sol.message)
    return sol


def predict(experiments,viscosity=TRUE_PARAMETER,tolerance=2e-9):
    x,w=leggauss(100);x=(x+1)/2;w=w/2
    out=[]
    for e in experiments:
        sol=solve_channel(e['half_gap'],e['concentration'],e['surface_charge'],viscosity,tolerance)
        scale=e['pressure_gradient']*(e['half_gap']*1e-6)**2/viscosity
        value=w@sol.sol(x)[2] if e['observable']=='mean' else sol.sol(e['y'])[2]
        out.append(scale*value)
    return np.array(out)


def reading(g,half_gap=.3,concentration=.01,surface_charge=0.,observable='mean',y=.5):
    return dict(pressure_gradient=float(g),half_gap=float(half_gap),concentration=float(concentration),
                surface_charge=float(surface_charge),observable=observable,y=float(y))


def calibration_inputs():
    return [reading(g,half_gap=h,observable=o,y=y) for h in [.25,.4]
            for o,y in [('mean',.5),('point',.2),('point',.7)] for g in np.linspace(-1e5,1e5,20)]


def hidden_inputs():
    return {name:[reading(g,h,c,q,o,y) for g in np.linspace(-9e4,9e4,12)
                  for o,y in [('mean',.5),('point',.2),('point',.65)]]
            for name,h,c,q in [('positive_charge',.22,.0065,.00032),
                               ('negative_charge',.38,.007,-.00034),
                               ('overlapping_layers',.24,.008,.00032)]}
