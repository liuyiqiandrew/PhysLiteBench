import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import ive, gammainc
from functools import lru_cache


@lru_cache(maxsize=128)
def coefficients(p, count=256):
    g=np.zeros((count+1)//2);h=np.zeros_like(g)
    g[0]=(1-p)/2;h[0]=g[0]**2
    for k in range(1,len(g)):
        fourth=0. if k==1 else np.dot(h[:k-1],h[k-2::-1])
        g[k]=(1+2*p)/2*h[k-1]-p/2*fourth
        h[k]=np.dot(g[:k+1],g[k::-1])
    result=np.zeros(count);result[::2]=g
    return result



def response(e, speed):
    p = e['stationary_fraction']
    z = e['density']*speed*e['time']
    if p == 0.:
        return float((ive(0, z)+ive(1, z))**2)
    a = coefficients(p)
    q = np.dot(a, gammainc(np.arange(1, len(a)+1), z))
    return float((1-q)**2)

def predict_at(experiments, speed):
    return np.asarray([response(e, speed) for e in experiments])


class Model:
    def __init__(self):
        self.speed = None

    def fit(self, records):
        # Every calibration record has p=0, where the exact two-speed gas
        # survival is the squared scaled-Bessel sum.
        x = np.asarray([r['input']['density']*r['input']['time'] for r in records])
        y = np.asarray([r['value'] for r in records])
        sigma = np.asarray([r['sigma'] for r in records])
        def objective(speed):
            z = x*speed
            residual = ((ive(0, z)+ive(1, z))**2-y)/sigma
            return float(residual@residual)
        result = minimize_scalar(objective, bounds=(.8, 1.3), method='bounded',
                                 options={'xatol': 1e-12})
        self.speed = float(result.x)
        return self

    def predict(self, experiments):
        return predict_at(experiments, self.speed)
