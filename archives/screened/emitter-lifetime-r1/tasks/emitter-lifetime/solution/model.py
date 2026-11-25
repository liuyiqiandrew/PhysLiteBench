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
        if epsilon_imag>0:
            def tail(v):
                q=np.sqrt(epsilon-1-v*v+0j)
                rs=(1j*v-q)/(1j*v+q);rp=(epsilon*1j*v-q)/(epsilon*1j*v+q)
                angular=(1+v*v)*rp if vertical else rs+v*v*rp
                return float(np.imag(angular)*np.exp(-2*height*v))
            value+=quad(tail,0.,np.inf,epsabs=2e-11,epsrel=2e-11,limit=200)[0]
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
        from scipy.optimize import minimize_scalar
        experiments=[r['input'] for r in records]
        x=np.array([-np.log(survival(e,1.))/e['time'] if e['time'] else 1. for e in experiments])*np.array([e['time'] for e in experiments])
        y=np.array([r['value'] for r in records]);sigma=np.array([r['sigma'] for r in records])
        result=minimize_scalar(lambda rate:np.sum(((np.exp(-rate*x)-y)/sigma)**2),bounds=(.4,1.2),method='bounded',options={'xatol':1e-13})
        self.vacuum_rate=float(result.x)
        return self

    def predict(self,experiments):return np.array([survival(e,self.vacuum_rate) for e in experiments])
