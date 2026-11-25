"""Independent species-amount and differential piston-volume balances."""
from functools import lru_cache
import numpy as np
from scipy.integrate import solve_ivp

TRUE_RATE=.11


@lru_cache(maxsize=128)
def trajectory(initial,left_fraction,tolerance=2e-12):
    volumes=np.array([left_fraction,1-left_fraction])
    c=np.array(initial).reshape(2,2)
    state=np.column_stack((c*volumes[:,None],volumes)).ravel()
    def rhs(time,state):
        state=state.reshape(2,3)
        c=state[:,:2]/state[:,2,None]
        reaction=state[:,2]*(c[:,0]-c[:,1]**2/.75)
        density=state[:,:2].sum()/state[:,2].sum()
        volume_rate=(reaction-state[:,2]/state[:,2].sum()*reaction.sum())/density
        return np.column_stack((-reaction,2*reaction,volume_rate)).ravel()
    return solve_ivp(rhs,(0.,24.),state,method='DOP853',max_step=.1,
                     rtol=tolerance,atol=tolerance*.01,dense_output=True).sol


def predict(experiments,rate=TRUE_RATE):
    out=[]
    for e in experiments:
        sol=trajectory(tuple(np.asarray(e['initial'],dtype=float).ravel()),e['left_fraction'])
        state=sol(rate*e['time']).reshape(2,3)
        out.append(state[e['chamber'],1]/state[e['chamber'],2])
    return np.array(out)


def readings(initial,fraction,times):
    return [dict(initial=np.asarray(initial).tolist(),left_fraction=fraction,time=float(t),chamber=j)
            for t in times for j in [0,1]]


def calibration_inputs():
    return sum([readings([c,c],fraction,[.25,1.,3.,7.,15.,35.])
                for c in [[.7,.3],[.1,.9],[1.2,.4],[.2,.3]]
                for fraction in [.3,.65]],[])


def hidden_inputs():
    return {name:sum([readings(c,fraction,[.4,1.5,4.,10.,25.,65.]) for fraction in [.25,.5,.75]],[])
            for name,c in dict(opposing=[[1.1,.1],[.1,1.1]],
                               asymmetric=[[1.8,.2],[.7,1.3]],
                               reverse=[[.08,.72],[.65,.15]]).items()}
