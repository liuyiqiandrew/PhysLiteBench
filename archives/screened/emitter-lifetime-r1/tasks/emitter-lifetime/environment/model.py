from functools import lru_cache
import numpy as np
from scipy.integrate import quad


@lru_cache(maxsize=1024)
def factors(height,epsilon_real,epsilon_imag):
    epsilon=complex(epsilon_real,epsilon_imag)
    if epsilon==1:return np.array([1.,1.])
    points=[np.sqrt(1-epsilon_real)] if epsilon_imag==0 and epsilon_real<1 else None
    values=[]
    for vertical in [True,False]:
        def kernel(w):
            q=np.sqrt(epsilon-1+w*w+0j)
            rs=(w-q)/(w+q);rp=(epsilon*w-q)/(epsilon*w+q)
            angular=(1-w*w)*rp if vertical else rs-w*w*rp
            return float(np.real(angular*np.exp(2j*height*w)))
        value=quad(kernel,0.,1.,points=points,epsabs=2e-11,epsrel=2e-11)[0]

        values.append(1+(1.5 if vertical else .75)*value)
    return np.array(values)


def survival(experiment,vacuum_rate):
    e=experiment
    components=factors(e['height'],e['epsilon_real'],e['epsilon_imag'])
    rate=components@np.array([np.cos(e['tilt'])**2,np.sin(e['tilt'])**2])
    return float(np.exp(-vacuum_rate*rate*e['time']))


class Model:
    def __init__(self):self.vacuum_rate=None

    def fit(self,records):
        raise NotImplementedError

    def predict(self,experiments):return np.array([survival(e,self.vacuum_rate) for e in experiments])
