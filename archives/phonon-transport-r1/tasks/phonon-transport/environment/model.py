import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize_scalar

ORDER=64
SPEED=1.


def predict_at(experiments,resistive_rate,order=ORDER):
    indices=np.arange(order-1)
    off=(indices+1)/np.sqrt((2*indices+1)*(2*indices+3))
    direction=np.diag(off,1)+np.diag(off,-1)
    out=[]
    for e in experiments:
        initial=np.zeros(order,dtype=complex)
        initial[:3]=np.array(e['initial'])/np.sqrt([1.,3.,5.])
        decay=np.full(order,resistive_rate+e['normal_rate'])
        decay[0]=0.
        if e['wavenumber']==0:
            final=np.exp(-decay*e['time'])*initial
        else:
            generator=-1j*SPEED*e['wavenumber']*direction-np.diag(decay)
            final=expm(generator*e['time'])@initial
        value=final[e['moment']]/np.sqrt(2*e['moment']+1)
        out.append(float(value.real if e['quadrature']=='cosine' else -value.imag))
    return np.array(out)


class Model:
    def __init__(self):
        self.resistive_rate=None

    def fit(self,records):
        raise NotImplementedError

    def predict(self,experiments):
        return predict_at(experiments,self.resistive_rate)
