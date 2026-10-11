import numpy as np
from scipy.special import roots_legendre
from functools import lru_cache

@lru_cache(maxsize=None)
def nodes(n):
 x,w=roots_legendre(n);return (x+1)/2,w/2


def pressure(gamma,count,ratio,order=96):
 L=np.log(ratio);x,w=nodes(order)
 if count==1:
  Z=L*np.dot(w,np.exp((2-gamma)*L*x))
  return -np.expm1(-2*L)*np.exp((2-gamma)*L)/(2*Z)
 d=L*x;theta=np.pi*x
 factor=(1-np.exp(-d))[:,None]**2+4*np.exp(-d)[:,None]*np.sin(theta[None,:]/2)**2
 angular=(factor**(gamma/2))@w
 s=(L-d)[:,None]*x[None,:]
 radial=((L-d)*np.exp((2-gamma)*d))*((np.exp((4-3*gamma)*s))@w)
 Z=L*np.dot(w,angular*radial)
 boundary=L*np.dot(w,angular*np.exp((4-3*gamma)*(L-d)+(2-gamma)*d))
 return -np.expm1(-2*L)*boundary/(2*count*Z)


TRUE_PARAMETER = 1.12


def experiment(multiplier=.4,count=1,ratio=3.,radius=1.):
    return dict(temperature_multiplier=multiplier,ion_count=count,
                radius_ratio=ratio,disk_radius=radius)


def calibration_inputs():
    return [experiment(u,1,h,a) for u in [.15,.4,.8,1.2]
            for h in [1.5,3.,8.] for a in [.7,1.6]]


def hidden_inputs():
    return {
        'shared_anchors': [experiment(u,1,h,a) for u in [1.7,2.5,3.]
                           for h in [1.25,2.,12.] for a in [.5,2.]],
        'moderate_coupling': [experiment(u,2,h,a) for u in [1.4,1.6,1.8]
                             for h in [3.,5.,8.] for a in [.5,2.]],
        'correlated_pair': [experiment(u,2,h,a) for u in [2.,2.2,2.4]
                           for h in [2.,3.,5.] for a in [.5,2.]],
        'strong_coupling': [experiment(u,2,h,a) for u in [2.6,2.8,3.]
                           for h in [2.,4.,8.] for a in [.5,2.]],
    }


def predict(experiments,coupling,order=96):
    return np.asarray([pressure(e['temperature_multiplier']*coupling,e['ion_count'],
                                e['radius_ratio'],order) for e in experiments])
