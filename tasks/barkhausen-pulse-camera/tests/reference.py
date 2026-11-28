"""Independent fixed-lifetime radial OU bridge; integrate over the other lifetime."""
from functools import lru_cache
import numpy as np
from scipy.integrate import trapezoid
from scipy.special import roots_legendre


def bridge(h,w,g,s,times,loads):
    v,k,scale=w
    D=g*scale
    om=np.sqrt((k/2)**2+D*s)
    den=-np.expm1(-2*om*h)
    ch=(1+np.exp(-2*om*h))/den
    flux=v/D*(2*om*np.exp(-om*h)/den)**2*np.exp(v/D*(k/2-om*ch))
    active=[(t,a) for t,a in zip(times,loads) if t<h and a>0]
    if not active:return flux
    def cov(t,u):
        lo,hi=sorted((t,u))
        return (-np.expm1(-2*om*lo))*(-np.expm1(-2*om*(h-hi)))*np.exp(-om*(hi-lo))/(2*om*den)
    mean=lambda t: np.sqrt(v)*np.exp(-om*t)*(-np.expm1(-2*om*(h-t)))/den
    t,a=active[0]
    A=1+D*a*cov(t,t)
    m=mean(t)
    if len(active)==1:return flux*np.exp(-a*m*m/A)/A**2
    (u,b)=active[1]
    B=1+D*b*cov(u,u)
    C=D*cov(t,u)
    n=mean(u)
    det=A*B-a*b*C*C
    quad=(a*B*m*m+b*A*n*n-2*a*b*C*m*n)/det
    return flux*np.exp(-quad)/det**2


@lru_cache(None)
def rule(n):
    return roots_legendre(n)


def integrate_other(w,g,s,times,loads,T,last,n):
    # Split at every flash and trigger. After the last trigger integrate the
    # tail on a logarithmic mesh so short early lifetimes retain resolution.
    if last:
        cuts=sorted(set([0.,.04,.12,*times,T]))
    else:
        cuts=[T,T+.4,T+1.2,T+3,T+8,T+16,T+38,T+72]
    x,ww=rule(n)
    ret=0.
    for lo,hi in zip(cuts,cuts[1:]):
        for h,weight in zip(lo+(x+1)*(hi-lo)/2,ww):
            ret+=weight*(hi-lo)/2*bridge(h,w,g,s,times,loads)
    return ret


def measure(q,g,n=14,frequency_step=.42,cutoff=660.):
    y=np.arange(0,cutoff,frequency_step)
    f=1.8+1j*y
    flashes=np.array(q["flashes"])
    T=q["time"]; walls=q["walls"]; ts=flashes[:,0]
    def inv(loads):
        F=0.
        indx=[0,1] if q["identity"]=="blind" else [q["identity"]]
        for i in indx:
            F+=bridge(T,walls[i],g,f,ts,loads[:,i])*integrate_other(
                walls[1-i],g,f,ts,loads[:,1-i],T,q["trigger"]=="last",n)
        vals=np.real(np.exp(1j*y*q["size"])*F)
        return trapezoid(vals,dx=frequency_step)
    load=flashes[:,1:]
    return inv(load)/inv(0*load)
