"""Killed interval modes after the Brownian-bridge change of coordinate."""
import numpy as np
from scipy.special import roots_legendre
from meter import windows

r,w=roots_legendre(36)
j=np.arange(1,80)
m=np.pi*j


def measure(q,D):
    a,b,g=windows(q)
    (u,k),(v,h)=q["walls"]
    low=lambda x:k*x-u
    width=lambda x:v-h*(q["length"]-x)-low(x)
    def clock(x):
        return (x-a)/(width(a)*width(x))
    def modes(x,lo,hi):
        lo=max(0.,(lo-low(x))/width(x))
        hi=min(1.,(hi-low(x))/width(x))
        p=(hi+lo)/2+(hi-lo)*r/2
        return np.sin(np.outer(m,p)),w*(hi-lo)/2
    def decay(t):
        return np.exp(-D*m*m*t)
    def run(intervals):
        # Entrance at the upper edge and absorption at the lower edge.
        c=(-1.)**(j+1)*m
        last=0.
        for x,lo,hi in sorted(intervals):
            t=clock(x)
            H,wt=modes(x,lo,hi)
            c=2*(H@(wt*((c*decay(t-last))@H)))
            last=t
        return np.sum(c*decay(clock(b)-last)*m)
    return run(g)/run(g[:1])
