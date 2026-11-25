import numpy as np
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
        raise NotImplementedError

    def predict(self,experiments):
        a,b=coefficients(experiments)
        t=np.array([e['wait'] for e in experiments])
        return np.clip(np.prod(a+b*np.exp(-self.gamma*t[:,None]),axis=1),0,1)
