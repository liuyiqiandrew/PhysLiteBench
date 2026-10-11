"""Dilute-limit free-energy minimization, backed by the finite-cell PB BVP."""
from functools import lru_cache
import numpy as np
from scipy.optimize import minimize
from scipy.integrate import solve_bvp

TRUE_PARAMETER = 1.3


def experiment(multiplier=.1,bjerrum=.8,radius=1.,fraction=.5):
    return dict(charge_multiplier=multiplier,bjerrum_length=bjerrum,
                rod_radius=radius,trivalent_charge_fraction=fraction)


def calibration_inputs():
    return [experiment(u,b,a,f) for u in [.1,.15] for b in [.6,.9,1.25]
            for a in [.7,1.6] for f in [0.,.3,.7,1.]]


def hidden_inputs():
    scales = [(u,b) for u,b in [(.85,.9),(.95,1.1),(1.,1.25)]]
    return {
        'shared_anchors': [experiment(u,b,a,f) for u,b in scales
                           for a in [.5,2.] for f in [0.,1.]],
        'selective_condensation': [experiment(u,b,a,.85) for u,b in scales
                                  for a in [.5,1.2,2.]],
        'trivalent_exhaustion': [experiment(u,b,a,.4) for u,b in scales
                                for a in [.5,1.2,2.]],
        'two_condensed_species': [experiment(u,b,a,.2) for u,b in [(1.,1.1),(.95,1.2),(1.,1.25)]
                                  for a in [.5,1.2,2.]],
    }


@lru_cache(maxsize=None)
def population(xi,fraction):
    # Length measured in Bjerrum units. nu are uncondensed numbers per length.
    # The coefficient of log(R/a) in beta*F is E=Q_free^2-2*N_free.
    valence = np.array([1.,3.])
    total = xi*np.array([1-fraction,fraction/3])
    def objective(nu):
        charge = valence@nu
        return charge*charge-2*np.sum(nu)
    def gradient(nu):
        return 2*(valence@nu)*valence-2
    result = minimize(objective,total,jac=gradient,bounds=list(zip(np.zeros(2),total)),
                      method='L-BFGS-B',options={'ftol':1e-15,'gtol':1e-12,'maxiter':1000})
    assert result.success,result.message
    return result.x


def predict(experiments,line_density):
    result = []
    for e in experiments:
        xi = e['bjerrum_length']*e['charge_multiplier']*line_density
        f = e['trivalent_charge_fraction']
        nu = population(xi,f)
        charge = nu[0]+3*nu[1]
        result.append((np.sum(nu)-charge*charge/2)/(xi*(1-f+f/3)))
    return np.asarray(result)


def finite(xi,f,z,length,previous=None):
    # w_i=log(4*pi*lB*r^2*n_i); E=d psi/d log r, psi=e Phi/kBT.
    s=np.unique(np.r_[np.linspace(0,1,401),np.geomspace(1e-5,.1,100),1-np.geomspace(1e-5,.1,100)])
    if previous is None:
        w1=np.full_like(s,np.log(2*xi*(1-f)/length))
        wh=np.full_like(s,np.log(2*xi*f/(z*length)))
        vals=np.array([w1,wh,2*xi*(s-1),2*xi*f*s])
    else:
        vals=previous.sol(s)
    def rhs(s,y):
        v1=np.exp(y[0]); vh=np.exp(y[1])
        return length*np.vstack([2+y[2],2+z*y[2],v1+z*vh,z*vh])
    def bc(left,right):
        return [left[2]+2*xi,right[2],left[3],right[3]-2*xi*f]
    sol=solve_bvp(rhs,bc,s,vals,tol=1e-7,max_nodes=30000)
    assert sol.success,sol.message
    phi=(np.exp(sol.y[0,-1])+np.exp(sol.y[1,-1]))*(-np.expm1(-2*length))/(4*xi*(1-f+f/z))
    return phi,sol


def solve(xi,f,z,length):
    sol=None
    for L in np.geomspace(.1,length,100):
        p,sol=finite(xi,f,z,L,sol)
    return p


