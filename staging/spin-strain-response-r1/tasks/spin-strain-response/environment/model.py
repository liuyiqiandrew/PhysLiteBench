import numpy as np
from scipy.linalg import eigh

SX=np.array([[0.,1.,0.],[1.,0.,1.],[0.,1.,0.]])/np.sqrt(2.)
SZ=np.diag([1.,0.,-1.])
Q=SZ@SZ


def response(e):
    h=e['anisotropy']*Q+e['transverse']*SX+e['longitudinal']*SZ
    energy,vectors=eigh(h)
    weights=np.exp(-(energy-energy[0])/e['temperature'])
    weights/=weights.sum()
    observable=vectors.T@Q@vectors
    mean=np.dot(weights,np.diag(observable))
    second=np.dot(weights,np.diag(observable@observable))
    return float((second-mean**2)/e['temperature'])


def predict_at(experiments,coupling):
    return coupling**2*np.array([response(e) for e in experiments])


class Model:
    def __init__(self):
        self.coupling=None

    def fit(self,records):
        raise NotImplementedError

    def predict(self,experiments):
        return predict_at(experiments,self.coupling)
