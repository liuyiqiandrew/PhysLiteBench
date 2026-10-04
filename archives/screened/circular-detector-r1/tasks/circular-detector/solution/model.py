from functools import lru_cache
import numpy as np
from scipy.integrate import quad


@lru_cache(maxsize=1024)
def circular_excitation(acceleration,speed,gap):
    gamma=1/np.sqrt(1-speed*speed)
    def regular(time):
        if time==0:return 1/12
        x=time/(2*gamma*speed)
        if abs(x)<.01:
            z=x*x
            difference=z*(1/3+z*(-2/45+z*(1/315-2*z/14175)))
        else:difference=1-np.sinc(x/np.pi)**2
        term=(gamma*speed)**2*difference
        return term/(time*time*(1+term))
    return acceleration*quad(regular,0,np.inf,weight='cos',wvar=gap/acceleration,epsabs=2e-11,limlst=300)[0]/(2*np.pi**2)


def response(experiment):
    a=experiment['acceleration'];gap=experiment['gap']
    if experiment['trajectory']=='cusp':excitation=a/(8*np.pi*np.sqrt(3))*np.exp(-2*np.sqrt(3)*gap/a)
    else:excitation=circular_excitation(a,experiment['speed'],gap)
    return float(excitation+(gap/(2*np.pi) if experiment['readout']=='deexcitation' else 0.))


class Model:
    def __init__(self):self.coupling=None

    def fit(self,records):
        basis=np.array([response(r['input']) for r in records])
        y=np.array([r['value'] for r in records]);sigma=np.array([r['sigma'] for r in records])
        self.coupling=float(np.sum(basis*y/sigma**2)/np.sum(basis*basis/sigma**2))
        return self

    def predict(self,experiments):return self.coupling*np.array([response(e) for e in experiments])
