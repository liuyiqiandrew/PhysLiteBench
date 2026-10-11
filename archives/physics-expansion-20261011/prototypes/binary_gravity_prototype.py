import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq, minimize_scalar
from functools import lru_cache

@lru_cache(maxsize=32768)
def integrate(alpha,ratio,central):
    eps=1e-5
    def rhs(x,y):
        psi,v,ml,mh=y
        light=(1-central)*np.exp(-psi);heavy=central*np.exp(-ratio*psi)
        return [v,light+heavy-2*v/x,x*x*light,x*x*heavy]
    out=solve_ivp(rhs,(eps,alpha),[eps**2/6,eps/3,(1-central)*eps**3/3,central*eps**3/3],rtol=2e-9,atol=2e-11,dense_output=True)
    return out

@lru_cache(maxsize=32768)
def equilibrium(alpha,ratio,fraction):
    def difference(central):
        out=integrate(alpha,ratio,central)
        ml,mh=out.y[2:,-1]
        return mh/(ml+mh)-fraction
    central=brentq(difference,1e-12,1-1e-12,xtol=1e-11)
    out=integrate(alpha,ratio,central)
    return alpha*out.y[1,-1],central,out

if __name__=='__main__':
    for ratio,fraction in [(2,.1),(2,.3),(3,.1),(3,.3),(4,.05),(4,.1),(4,.3),(6,.05),(6,.1),(6,.2)]:
        grid=np.linspace(.5,20,50)
        eta_values=[equilibrium(a,ratio,fraction)[0] for a in grid]
        peak=next(i for i in range(1,len(grid)-1) if eta_values[i-1]<eta_values[i]>eta_values[i+1])
        sol=minimize_scalar(lambda a:-equilibrium(a,ratio,fraction)[0],bounds=(grid[peak-1],grid[peak+1]),method='bounded')
        a=sol.x;eta,central,out=equilibrium(a,ratio,fraction)
        coeff=1/eta
        approx=.397211365783/(1-fraction+fraction/ratio)
        print(ratio,fraction,'alpha',a,'eta',eta,'central',central,'theta',coeff,'approx',approx,'error',approx/coeff-1,flush=True)
