"""Renewal propagation of the scaled depth distribution after a temperature change."""
from functools import lru_cache
import numpy as np
from scipy.special import roots_legendre, gammaln, hyp1f1

@lru_cache(None)
def nodes(order):
    return roots_legendre(order)

def fresh(v,a,t):
    return (np.sin(np.pi*a)/(np.pi*a))*np.exp(a*np.log(t)-gammaln(a+1))*hyp1f1(1,1+a,-t*v**(1/a))

def advance(w,v,mu,a,duration,order=100):
    rate=v**(1/a)
    z,c=nodes(order)
    # The first renewal may occur at any time in this interval.
    t=(1+z)/2
    left=t**8
    right=(1-t)**8
    total=left+right
    time=duration*left/total
    ages=(duration*right/total)[:,None]
    dt=c*4*duration*(t*(1-t))**7/total**2
    first=np.exp(-time[:,None]*rate)@(mu*rate)
    return mu*np.exp(-duration*rate)+w*((dt*first)@fresh(v[None,:],a,ages))

def line(q,Tg,order=700):
    z,w=nodes(order); v=np.exp(55*z-5);w=w*55*v
    ts=q['temperatures']; a=ts[0]/Tg
    A,D=q['write'];rate=v**(1/a)
    mu=w*fresh(v,a,A)*np.exp(-D*rate)
    def final(lock):
        m=mu.copy()
        for i,(t,g) in enumerate(q['pulses']):
            b=ts[i+1]/Tg
            rate=v**(1/b)
            if lock and i>=q['read'][0]:
                frac=q['read'][1] if i==q['read'][0] else 0.
            else:
                frac=1.
            if frac>0:m=advance(w,v,m,b,frac*t)
            m=m*np.exp(-((1-frac)*t+g)*rate)
        return m@np.exp(-q['recovery']*v**(1/a))
    p=final(False)
    return final(True)/p, p
