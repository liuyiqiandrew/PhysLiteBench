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
    temperature=e['temperature']
    result=(np.dot(weights,np.diag(observable)**2)-mean**2)/temperature
    for i in range(3):
        for j in range(i+1,3):
            gap=energy[j]-energy[i]
            quotient=weights[i]/temperature if gap==0. else weights[i]*(-np.expm1(-gap/temperature))/gap
            result+=2.*quotient*observable[i,j]**2
    return float(result)


def predict_at(experiments,coupling):
    return coupling**2*np.array([response(e) for e in experiments])


class Model:
    def __init__(self):
        self.coupling=None

    def fit(self,records):
        x=predict_at([r['input'] for r in records],1.)
        y=np.array([r['value'] for r in records])
        sigma=np.array([r['sigma'] for r in records])
        a=x/sigma; b=y/sigma
        self.coupling=float(np.sqrt(np.clip(np.dot(a,b)/np.dot(a,a),.8**2,1.4**2)))
        return self

    def predict(self,experiments):
        return predict_at(experiments,self.coupling)
