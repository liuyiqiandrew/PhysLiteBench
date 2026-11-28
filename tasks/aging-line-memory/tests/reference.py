"""Time evolution from the depth-resolved complete-graph master equation."""
import numpy as np
from scipy.special import roots_legendre

def evolve(weight,v,w,shape,t,degree=24):
    rate=v**(1/shape)
    j=np.arange(degree)
    theta=j*np.pi/degree
    cot=1/np.tan(theta[1:])
    delta=np.r_[.4*degree,.4*degree*theta[1:]*(cot+1j)]
    coeff=np.r_[.5,np.ones(degree-1)+1j*theta[1:]*(1+cot**2)-1j*cot]*np.exp(delta)*.4/t
    s=(delta/t)[:,None]
    denom=s+rate
    if weight is None:
        incoming=1
    else:
        incoming=(rate/denom)@weight
        incoming=incoming[:,None]
    recycle=(np.sin(np.pi*shape)/(np.pi*shape))*s**(-shape)*incoming/denom
    out=w*(coeff@recycle).real
    if weight is not None:
        out+=weight*np.exp(-rate*t)
    return out


def reading(q,Tg,order=700):
    z,w=roots_legendre(order);v=np.exp(44*z-6);w=w*44*v
    ts=q['temperatures'];a=ts[0]/Tg;A,D=q['write']
    m=evolve(None,v,w,a,A)*np.exp(-D*v**(1/a))
    def run(lock):
        p=m.copy()
        for i,(dt,keep) in enumerate(q['pulses']):
            b=ts[i+1]/Tg
            if lock and i>=q['read'][0]:
                f=q['read'][1] if i==q['read'][0] else 0
            else:f=1
            if f:p=evolve(p,v,w,b,f*dt)
            p*=np.exp(-((1-f)*dt+keep)*v**(1/b))
        return p@np.exp(-q['recovery']*v**(1/a))
    p=run(False)
    return run(True)/p, p
