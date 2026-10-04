"""Single-channel probe FCS from uniform phase-averaged coherent statistics."""
from functools import lru_cache
import numpy as np

PARAMETER='mixing_angle'
TRUE_PARAMETER=.51


def experiment(bias=1.,beta=.5,observable='third_cumulant',lead=0,other=0):
    return dict(bias=float(bias),beta=float(beta),observable=observable,lead=int(lead),other=int(other))


def calibration_inputs():
    return [experiment(v,b,observable,lead,other) for b in [.2,.4,.65,.78]
            for v in [-1.2,-.4,.4,1.2]
            for observable,lead,other in [('current',0,0),('current',1,1),('noise',0,0),('noise',1,1),('noise',0,1)]]


def hidden_inputs():
    return {name:[experiment(v,b,lead=lead) for b in betas for v in [-1.3,-.6,.5,1.2] for lead in [0,1]]
            for name,betas in [('weak_contact',[.3,.35,.4,.45]),('intermediate_contact',[.48,.52,.56,.60]),('strong_contact',[.67,.71,.75,.79])]}


@lru_cache(2048)
def cumulants(alpha,beta,points=1024):
    # Close the probe channel with a lossless one-channel reflector, then
    # average the coherent log-generating function uniformly over its phase.
    components=np.array([np.cos(alpha)*np.cos(beta),np.sin(alpha)*np.cos(beta),np.sin(beta)])
    matrix=np.eye(3)-2*components[:,None]*components[None,:]
    phase=2*np.pi*(np.arange(points)+.5)/points
    reflector=np.exp(1j*phase)
    amplitude=matrix[1,0]+matrix[1,2]*reflector*matrix[2,0]/(1-matrix[2,2]*reflector)
    probability=np.abs(amplitude)**2
    return np.array([np.mean(probability),np.mean(probability*(1-probability)),
                     np.mean(probability*(1-probability)*(1-2*probability))])


def predict(experiments,alpha=TRUE_PARAMETER):
    out=[]
    for e in experiments:
        coefficients=cumulants(float(alpha),e['beta'])
        sign=1 if e['lead']==0 else -1
        if e['observable']=='current':
            value=e['bias']*sign*coefficients[0]
        elif e['observable']=='noise':
            other=1 if e['other']==0 else -1
            value=abs(e['bias'])*sign*other*coefficients[1]
        else:
            value=e['bias']*sign*coefficients[2]
        out.append(float(value))
    return np.array(out)
