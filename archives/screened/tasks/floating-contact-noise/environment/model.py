import numpy as np
from scipy.optimize import minimize_scalar


def scattering(alpha,beta):
    u=np.array([np.cos(alpha)*np.cos(beta),np.sin(alpha)*np.cos(beta),np.sin(beta)])
    return np.eye(3)-2*np.outer(u,u)


def dc_state(e,alpha):
    s=scattering(alpha,e['beta'])
    conductance=np.eye(3)-np.abs(s)**2
    potential=np.array([e['bias'],0.,0.])
    potential[2]=-conductance[2,:2]@potential[:2]/conductance[2,2]
    return s,conductance,potential,conductance@potential


def current_covariance(s,potential):
    # Cov(Q)/time for voltage-clamped Fermi reservoirs at temperature zero.
    a=np.array([np.diag(np.eye(3)[lead])-np.outer(s[lead].conj(),s[lead]) for lead in range(3)])
    occupied_window=np.maximum(potential[:,None]-potential[None,:],0.)
    return np.einsum('agd,bdg,gd->ab',a,a,occupied_window).real


def floating_covariance(s,g,mu):
    covariance=current_covariance(s,mu)
    response=np.eye(3)
    response[:,2]-=g[:,2]/g[2,2]
    return response@covariance@response.T


def third_cumulant(s,g,mu,lead):
    # Transfer statistics at the stationary contact potential, with the same
    # charge-response map used for the second cumulant.
    from itertools import combinations
    response=np.eye(3)[lead].copy()
    response[2]-=g[lead,2]/g[2,2]
    total=0.
    edges=np.unique(mu)
    for low,high in zip(edges[:-1],edges[1:]):
        occupied=np.flatnonzero(mu>(low+high)/2)
        initial=np.isin(np.arange(3),occupied).astype(float)
        probabilities=[];transfers=[]
        for output in combinations(range(3),len(occupied)):
            probabilities.append(abs(np.linalg.det(s[np.ix_(output,occupied)]))**2)
            transfers.append(response@(initial-np.isin(np.arange(3),output)))
        probabilities=np.array(probabilities);transfers=np.array(transfers)
        mean=probabilities@transfers
        total+=(high-low)*(probabilities@(transfers-mean)**3)
    return float(total)


def predict_at(experiments,alpha):
    out=[]
    for e in experiments:
        s,g,mu,current=dc_state(e,alpha)
        if e['observable']=='current':
            value=current[e['lead']]
        elif e['observable']=='noise':
            value=floating_covariance(s,g,mu)[e['lead'],e['other']]
        else:
            value=third_cumulant(s,g,mu,e['lead'])
        out.append(float(value))
    return np.array(out)


class Model:
    def __init__(self):
        self.mixing_angle=None

    def fit(self,records):
        raise NotImplementedError

    def predict(self,experiments):
        return predict_at(experiments,self.mixing_angle)
