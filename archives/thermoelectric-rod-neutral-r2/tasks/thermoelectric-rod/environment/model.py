import numpy as np
from scipy.integrate import solve_ivp
from scipy.sparse import diags

HEAT_CAPACITY=1.5e6
RESISTIVITY=2e-5
SEEBECK_300=2e-4
SEEBECK_SLOPE=4e-6


def terminal_voltage(temperature,current,x,seebeck_span):
    left,right=temperature[:,0],temperature[:,-1]
    mean=np.trapezoid(temperature,x,axis=1)/(x[-1]-x[0])
    return (RESISTIVITY*current*(x[-1]-x[0])+SEEBECK_300*(right-left)
            +.5*SEEBECK_SLOPE*((right-300.)**2-(left-300.)**2)
            +seebeck_span*(.5*(right+left)-mean))


def thermal_solution(t,x,initial,current,boundary,seebeck_span,conductivity,refinement=2):
    fine=np.linspace(x[0],x[-1],refinement*(len(x)-1)+1)
    dx=fine[1]-fine[0];start=np.interp(fine,x,initial)
    if t[-1]==0:return initial[None].copy()
    def derivative(time,interior):
        temperature=np.r_[boundary[0],interior,boundary[1]]
        curvature=(temperature[2:]-2*interior+temperature[:-2])/dx**2
        gradient=(temperature[2:]-temperature[:-2])/(2*dx)
        return (conductivity*curvature+RESISTIVITY*current**2
                -SEEBECK_SLOPE*current*interior*gradient)/HEAT_CAPACITY
    n=len(start)-2
    pattern=diags([np.ones(n-abs(k)) for k in [-1,0,1]],[-1,0,1])
    answer=solve_ivp(derivative,(0.,t[-1]),start[1:-1],t_eval=t,method='BDF',
                     rtol=2e-9,atol=1e-7,jac_sparsity=pattern)
    if not answer.success:raise RuntimeError(answer.message)
    return np.column_stack([np.full(len(t),boundary[0]),answer.y.T,np.full(len(t),boundary[1])])[:,::refinement]


class Model:
    def __init__(self):self.conductivity=None

    def fit(self,data):
        raise NotImplementedError

    def predict(self,t,x,initial,current,boundary,seebeck_span):
        t,x,initial=np.asarray(t),np.asarray(x),np.asarray(initial)
        temperature=thermal_solution(t,x,initial,current,boundary,seebeck_span,self.conductivity)
        return {'temperature':temperature,'voltage':terminal_voltage(temperature,current,x,seebeck_span)}
