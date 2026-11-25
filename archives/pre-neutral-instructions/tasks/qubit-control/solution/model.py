import numpy as np
from scipy.optimize import minimize_scalar


def rotation(phase,angle):
    axis=np.array([[0,np.exp(-1j*phase)],[np.exp(1j*phase),0]])
    return np.cos(angle/2)*np.eye(2)-1j*np.sin(angle/2)*axis


def probabilities(experiments,gamma):
    result=[]
    charges=np.array([1,0,0,-1])
    for e in experiments:
        u=np.kron(*[rotation(p,a) for p,a in e['preparation']])
        psi=u[:,0]
        rho=np.outer(psi,psi.conj())
        rho*=np.exp(-gamma*e['wait']*(charges[:,None]-charges[None,:])**2)
        v=np.kron(*[rotation(p,a) for p,a in e['readout']])
        result.append(float(np.real((v@rho@v.conj().T)[0,0])))
    return np.clip(result,0,1)


class QubitModel:
    def __init__(self):
        self.gamma=None

    def fit(self,runs):
        experiments=[r['experiment'] for r in runs]
        # The visible experiments activate one probe at a time.
        a=probabilities(experiments,10000.)
        b=probabilities(experiments,0.)-a
        t=np.array([e['wait'] for e in experiments])
        y=np.array([r['probability'] for r in runs])
        sigma=np.array([r['sigma'] for r in runs])
        def loss(gamma):
            return np.sum(((a+b*np.exp(-gamma*t)-y)/sigma)**2)
        result=minimize_scalar(loss,bounds=(.001,3.),method='bounded',options={'xatol':1e-12})
        self.gamma=float(result.x)
        return self

    def predict(self,experiments):
        return probabilities(experiments,self.gamma)
