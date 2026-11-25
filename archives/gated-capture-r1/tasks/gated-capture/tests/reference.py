"""Coupled radial diffusion BVP and outer-reservoir extrapolation."""
from functools import lru_cache
import numpy as np
from scipy.integrate import solve_bvp

TRUE_PARAMETER = .95
SIGMA = .01


def finite_shell(D,a,kappa0,kappa1,k01,k10,R,tolerance=2e-9):
    probability = np.array([k10,k01])/(k01+k10)
    reactivity = np.array([kappa0,kappa1])
    transition = np.array([[-k01,k10],[k01,-k10]])
    radii = np.geomspace(a,R,240)
    def rhs(r,y):
        return np.vstack([y[2:]/(D*r*r),-r*r*(transition@y[:2])])
    def boundary(inner,outer):
        return np.r_[inner[2:]-a*a*reactivity*inner[:2],outer[:2]-probability]
    guess = np.vstack([probability[:,None]*(1-.4*a/radii),
                       np.repeat((probability*.4*a*D)[:,None],len(radii),axis=1)])
    solution = solve_bvp(rhs,boundary,radii,guess,tol=tolerance,bc_tol=tolerance,max_nodes=15000)
    if not solution.success:
        raise RuntimeError(solution.message)
    inner = solution.sol(a)
    outer = solution.sol(R)
    reaction = 4*np.pi*a*a*np.dot(reactivity,inner[:2])
    incoming = 4*np.pi*outer[2:].sum()
    return float(reaction), {'outer_flux':float(incoming),'minimum_density':float(solution.y[:2].min()),
                             'balance_error':float(abs(reaction-incoming)),'nodes':int(solution.x.size)}


@lru_cache(maxsize=2048)
def capture_rate(D,a,kappa0,kappa1,k01,k10,refinement=1):
    inverse_length = np.sqrt((k01+k10)/D)
    first = max(80*a,a+18/inverse_length)*refinement
    radii = first*np.array([1,2,4])
    fluxes = [finite_shell(D,a,kappa0,kappa1,k01,k10,R)[0] for R in radii]
    return float(np.polynomial.polynomial.polyfit(1/radii,fluxes,2)[0])


def predict(experiments,diffusivity):
    return np.asarray([capture_rate(diffusivity,*[e[k] for k in
                       ('radius','reactivity0','reactivity1','rate01','rate10')])
                       for e in experiments])


def calibration_inputs():
    return [dict(radius=a,reactivity0=k,reactivity1=k,rate01=u,rate10=v)
            for a in [.6,1.,1.7] for k in [.5,1.5,5.]
            for u,v in [(.04,.2),(1.,.3)]]*16


def hidden_inputs():
    def rows(items):
        return [dict(zip(('radius','reactivity0','reactivity1','rate01','rate10'),x)) for x in items]
    return {
        'slow_switching':rows([(.6,.05,8.,.04,.12),(1.,0.,10.,.1,.1),
                               (1.7,.2,5.,.06,.2),(1.4,6.,.1,.3,.07)]),
        'biased_gate':rows([(.8,.1,15.,.05,.25),(1.2,12.,.1,.3,.05),
                            (1.5,.05,7.,.04,.4),(1.,8.,.2,.6,.05)]),
        'mixed_geometry':rows([(.7,.1,9.,.2,.3),(1.8,10.,.1,.15,.15),
                                (1.3,.1,12.,.4,.5),(1.6,6.,.05,.35,.2)]),
        'equal_reactivity_anchors':rows([(.8,.7,.7,.05,1.3),(1.4,2.,2.,1.2,.08),
                                         (1.8,7.,7.,.3,.8),(.6,1.,1.,.4,.4)])
    }
