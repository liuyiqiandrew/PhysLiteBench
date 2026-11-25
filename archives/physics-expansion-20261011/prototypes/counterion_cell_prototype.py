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


def limiting(xi,length=400.):
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

if __name__=='__main__':
    for xi in [.084,.4,.8,.95,.999,1.,1.001,1.05,1.25,1.625,1.75]:
        exact=1-xi/2 if xi<=1 else 1/(2*xi)
        val=limiting(xi);fine=limiting(xi,800)
        direct,charge=bvp(xi,12.)
        print(xi,'limit error',val-exact,'refinement',fine-val,'BVP error',direct-finite_cell(xi,12)[0],'charge',charge,flush=True)
