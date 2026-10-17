import numpy as np
from scipy.special import roots_legendre
from functools import lru_cache

@lru_cache(maxsize=None)
def nodes(n):
 x,w=roots_legendre(n);return (x+1)/2,w/2


def few_ions(gamma,count,ratio,order=96):
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
    return [experiment(u,n,h,a) for u in [.3,.6,.9,1.2] for n in [1,2]
            for h in [1.5,3.,8.] for a in [.7,1.6]]


def hidden_inputs():
    return {
        'shared_anchors': [experiment(u,n,h,a) for u in [1.7,2.6] for n in [1,2]
                           for h in [1.25,12.] for a in [.5,2.]],
        'three_ion_loop': [experiment(u,3,h,a) for u in [1.8,2.,2.2]
                           for h in [3.,5.,8.] for a in [.5,2.]],
        'strong_loop': [experiment(u,3,h,a) for u in [2.4,2.6,2.8]
                        for h in [3.,5.,8.] for a in [.5,2.]],
        'wide_cell': [experiment(u,3,h,a) for u in [2.5,2.75,3.]
                      for h in [6.,9.,12.] for a in [.5,2.]],
    }


@lru_cache(maxsize=None)
def pressure(gamma,count,ratio,order=40,angles=160):
    if count < 3:
        return few_ions(gamma,count,ratio,max(96,order))
    length = np.log(ratio)
    x,w = nodes(order)
    angle,v = nodes(angles)
    theta = 2*np.pi*angle
    difference = np.sin((theta[:,None]-theta[None,:])/2)**2
    single = np.sin(theta/2)**2
    partition,boundary = 0.,0.
    for d1,w1 in zip(length*x,length*w):
        d2 = (length-d1)*x
        decay1 = np.exp(-d1)
        first = ((1-decay1)**2+4*decay1*single)**(gamma/2)
        decay2 = np.exp(-d2)[:,None,None]
        second = ((1-decay2)**2+4*decay2*difference)**(gamma/2)
        decay3 = np.exp(-d1-d2)[:,None]
        third = ((1-decay3)**2+4*decay3*single)**(gamma/2)
        angular = np.einsum('i,rij,rj,j->r',v*first,second,third,v,optimize=True)
        span = length-d1-d2
        s = span[:,None]*x[None,:]
        radial = span*(np.exp((6-6*gamma)*s)@w)
        weight = w1*(length-d1)*w*angular*np.exp((4-3*gamma)*d1+(2-gamma)*d2)
        partition += np.dot(weight,radial)
        boundary += np.dot(weight,np.exp((6-6*gamma)*span))
    return -np.expm1(-2*length)*boundary/(6*partition)


def predict(experiments,coupling,order=40,angles=160):
    return np.asarray([pressure(e['temperature_multiplier']*coupling,e['ion_count'],
                                e['radius_ratio'],order,angles) for e in experiments])
