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


class Model:
    def __init__(self):
        self.mixing_angle=None

    def fit(self,records):
        raise NotImplementedError

    def predict(self,experiments):
        out=[]
        for e in experiments:
            s,g,mu,current=dc_state(e,self.mixing_angle)
            if e['observable']=='current':
                value=current[e['lead']]
            else:
                covariance=current_covariance(s,mu)

                value=covariance[e['lead'],e['other']]
            out.append(float(value))
        return np.array(out)
