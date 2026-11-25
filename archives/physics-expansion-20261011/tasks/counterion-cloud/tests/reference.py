"""Independent finite-cell Poisson-Boltzmann pressure and dilution limit."""
import numpy as np
from scipy.optimize import brentq
from scipy.integrate import solve_bvp,simpson


def finite_cell(xi,length):
    crossover=length/(length+1)
    if abs(xi-crossover)<1e-14:
        constant=0.
    elif xi>crossover:
        def equation(beta):
            if xi<1:
                angle=np.arctan(xi*beta/(1-xi+beta*beta))
            else:
                angle=np.pi-np.arctan(beta)-np.arctan2(beta,xi-1)
            return angle/beta-length
        beta=brentq(equation,1e-12,np.pi/length,xtol=1e-14)
        constant=-4*beta*beta
    else:
        b=1-xi
        def equation(z):
            a=b*np.tanh(z)
            return 1-np.arctanh(a)/z-length*a/z
        z=brentq(equation,1e-8,length*b+np.arctanh(b)+1,xtol=1e-12)
        a=b*np.tanh(z)
        constant=4*a*a
    return (1-constant/4)*(-np.expm1(-2*length))/(2*xi),constant


def limiting(xi,length=800.):
    lengths=length*np.array([1.,2.,4.,8.])
    values=np.array([finite_cell(xi,L)[0] for L in lengths])
    # Extrapolate the finite-cell boundary pressure, without using a
    # condensation threshold or either infinite-cell response branch.
    scale=length/lengths
    matrix=np.column_stack([np.ones(4),scale**2,scale**3,scale**4])
    return np.linalg.solve(matrix,values)[0]


def bvp(xi,length):
    # u=log(4*pi*l_B*r^2*n(r)); u_tt=exp(u), prescribed electric fields.
    previous=None
    for L in np.geomspace(.25,length,12):
        s=np.linspace(0,1,201)
        if previous is None:
            vals=np.array([np.full_like(s,np.log(2*xi/L)),2*L*(1-xi)+2*xi*L*s])
        else:
            vals=previous.sol(s)
            vals[1]*=L/old_L
        def rhs(s,y):return np.vstack([y[1],L*L*np.exp(y[0])])
        def bc(left,right):return [left[1]-2*L*(1-xi),right[1]-2*L]
        previous=solve_bvp(rhs,bc,s,vals,tol=2e-8,max_nodes=10000)
        assert previous.success,previous.message
        old_L=L
    pressure=np.exp(previous.y[0,-1])*(-np.expm1(-2*length))/(4*xi)
    s=np.linspace(0,1,10001)
    charge=simpson(np.exp(previous.sol(s)[0]),x=s)*length/(2*xi)
    return pressure,charge


TRUE_PARAMETER = 1.3


def experiment(multiplier=.3,bjerrum=.8,radius=1.):
    return dict(charge_multiplier=multiplier,bjerrum_length=bjerrum,rod_radius=radius)


def calibration_inputs():
    return [experiment(u,b,a) for u in [.2,.3,.4]
            for b in [.6,.8,1.,1.25] for a in [.7,1.6]]


def hidden_inputs():
    return {
        'shared_anchors': [experiment(u,b,a) for u,b in [(.25,.7),(.5,1.1)]
                           for a in [.5,1.2,2.]],
        'strong_charge': [experiment(u,b,a) for u,b in [(.94,1.12),(.97,1.12),(1.,1.12)]
                          for a in [.5,1.2,2.]],
        'long_bjerrum_length': [experiment(u,1.2,a) for u in [.92,.96,1.]
                               for a in [.5,1.2,2.]],
        'largest_coupling': [experiment(u,1.25,a) for u in [.94,.97,1.]
                            for a in [.5,1.2,2.]],
    }


def predict(experiments,line_density,length=800.):
    return np.asarray([limiting(e['charge_multiplier']*e['bjerrum_length']*
                                line_density,length) for e in experiments])
