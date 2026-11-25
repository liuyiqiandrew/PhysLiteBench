"""Independent outgoing Slater-state charge counting at each energy window."""
from itertools import combinations
import numpy as np

PARAMETER='mixing_angle'
TRUE_PARAMETER=.51


def experiment(bias=1.,beta=.5,observable='noise',lead=0,other=0):
    return dict(bias=float(bias),beta=float(beta),observable=observable,lead=int(lead),other=int(other))


def calibration_inputs():
    return [experiment(v,b,'current',lead,lead) for b in [.2,.4,.65,.78] for v in np.linspace(-1.4,1.4,13) if abs(v)>.01 for lead in [0,1]]


def hidden_inputs():
    return {'source_noise':[experiment(v,b) for b in [.25,.45,.65,.8] for v in [.3,.6,1.,1.4]],
            'drain_noise':[experiment(v,b,lead=1,other=1) for b in [.3,.5,.7,.8] for v in [-1.3,-.7,.4,1.2]],
            'cross_noise':[experiment(v,b,lead=0,other=1) for b in [.2,.4,.6,.78] for v in [-1.4,-.6,.7,1.3]]}


def fixed_potential_statistics(e,alpha):
    weights=np.array([np.cos(alpha)**2*np.cos(e['beta'])**2,np.sin(alpha)**2*np.cos(e['beta'])**2,np.sin(e['beta'])**2])
    s=np.eye(3)-2*np.sqrt(weights[:,None]*weights[None,:])
    g=np.eye(3)-s*s
    mu=np.array([e['bias'],0.,weights[0]/(weights[0]+weights[1])*e['bias']])
    edges=np.unique(mu)
    total_mean=np.zeros(3);total_covariance=np.zeros((3,3))
    for low,high in zip(edges[:-1],edges[1:]):
        occupied=np.flatnonzero(mu>(low+high)/2)
        initial=np.isin(np.arange(3),occupied).astype(float)
        first=np.zeros(3);second=np.zeros((3,3));probability_sum=0.
        for output in combinations(range(3),len(occupied)):
            probability=float(abs(np.linalg.det(s[np.ix_(output,occupied)]))**2)
            transferred=initial-np.isin(np.arange(3),output).astype(float)
            first+=probability*transferred
            second+=probability*np.outer(transferred,transferred)
            probability_sum+=probability
        assert abs(probability_sum-1)<1e-12
        total_mean+=(high-low)*first
        total_covariance+=(high-low)*(second-np.outer(first,first))
    return total_mean,total_covariance,g


def predict(experiments,alpha):
    out=[]
    for e in experiments:
        mean,noise,g=fixed_potential_statistics(e,alpha)
        a,b=e['lead'],e['other']
        if e['observable']=='current':
            value=mean[a]
        else:
            # Integrate finite-capacitance island continuity over a long counting
            # interval. The bounded island charge contributes no variance/time.
            ra,rb=g[a,2]/g[2,2],g[b,2]/g[2,2]
            value=noise[a,b]-ra*noise[2,b]-rb*noise[a,2]+ra*rb*noise[2,2]
        out.append(float(value))
    return np.array(out)
