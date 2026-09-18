import numpy as np
from scipy.optimize import minimize_scalar
from scipy.spatial.transform import Rotation


def coefficients(experiments):
    a,b=[],[]
    for experiment in experiments:
        aa,bb=[],[]
        for (phase,angle),(rp,ra) in zip(experiment['preparation'],experiment['readout']):
            v=Rotation.from_rotvec(angle*np.array([np.cos(phase),np.sin(phase),0.])).apply([0.,0.,1.])
            w=Rotation.from_rotvec(-ra*np.array([np.cos(rp),np.sin(rp),0.])).apply([0.,0.,1.])
            aa.append((1+w[2]*v[2])/2)
            bb.append(np.dot(w[:2],v[:2])/2)
        a.append(aa);b.append(bb)
    return np.array(a),np.array(b)


class QubitModel:
    def __init__(self):
        self.gamma=None

    def fit(self,runs):
        experiments=[r['experiment'] for r in runs]
        a,b=coefficients(experiments)
        t=np.array([e['wait'] for e in experiments])
        y=np.array([r['probability'] for r in runs])
        sigma=np.array([r['sigma'] for r in runs])
        def loss(gamma):
            p=np.prod(a+b*np.exp(-gamma*t[:,None]),axis=1)
            return np.sum(((p-y)/sigma)**2)
        result=minimize_scalar(loss,bounds=(.001,3.),method='bounded',options={'xatol':1e-12})
        self.gamma=float(result.x)
        return self

    def predict(self,experiments):
        a,b=coefficients(experiments)
        t=np.array([e['wait'] for e in experiments])
        return np.clip(np.prod(a+b*np.exp(-self.gamma*t[:,None]),axis=1),0,1)
