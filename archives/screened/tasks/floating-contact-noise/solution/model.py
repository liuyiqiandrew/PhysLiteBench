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
    transmission=np.abs(s)**2
    b=transmission[0,1]
    d=4*transmission[2,0]*transmission[2,1]
    r=transmission[2,0]+transmission[2,1]
    # Derivatives at zero of the stationary voltage-probe generating factor.
    first=b+d/(4*r)
    second=first-d*d/(8*r**3)
    third=first-3*d*d/(8*r**3)+3*d**3/(16*r**5)
    cumulant=third-3*second*first+2*first**3
    return float(mu[0]*cumulant*(1 if lead==0 else -1))


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
        inputs=[r['input'] for r in records]
        values=np.array([r['value'] for r in records])
        sigma=np.array([r['sigma'] for r in records])
        def objective(alpha):
            return float(np.sum(((predict_at(inputs,alpha)-values)/sigma)**2))
        self.mixing_angle=float(minimize_scalar(objective,bounds=(.25,.7),method='bounded',options={'xatol':1e-12}).x)
        return self

    def predict(self,experiments):
        return predict_at(experiments,self.mixing_angle)
