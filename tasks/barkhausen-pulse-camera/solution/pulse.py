import numpy as np
from scipy.integrate import trapezoid


def single(wall,gain,s,T,times,load):
    v,k,scale=wall
    D=gain*scale
    root=np.sqrt(k*k+4*D*s)
    low=2*s/(root+k)
    gap=root/D
    def propagate(u,dt):
        q=np.exp(-root*dt)
        den=1+(u-low)*(-np.expm1(-root*dt))/gap
        return low+(u-low)*q/den, q/den**2
    # Infinite terminal Laplace load selects an already extinct wall.
    dt=T-times[-1]
    em=np.exp(-root*dt)
    u=low+gap*em/(1-em)
    ut=-gap*root*em/(1-em)**2
    # The stable root includes all motion after the final flash for an unrestricted wall.
    free=low
    for j in range(len(times)-1,-1,-1):
        dt=times[j]-(times[j-1] if j else 0)
        u,sensitivity=propagate(u+load[j],dt)
        ut*=sensitivity
        free,_=propagate(free+load[j],dt)
    dead=np.exp(-v*u)
    return dead,-v*ut*dead,np.exp(-v*free)


def transform(q,g,s,load):
    flashes=np.array(q["flashes"])
    law=[single(w,g,s,q["time"],flashes[:,0],load[:,i]) for i,w in enumerate(q["walls"])]
    idx=(0,1) if q["identity"]=="blind" else (q["identity"],)
    result=0
    for i in idx:
        dead,_,free=law[1-i]
        other=dead if q["trigger"]=="last" else free-dead
        result+=law[i][1]*other
    return result


def measure(q,g):
    S=q["size"]
    h=.25
    freq=np.arange(0,1000.+h,h)
    s=1.6+1j*freq
    phase=np.exp(1j*freq*S)
    load=np.array(q["flashes"])[:,1:]
    def invert(weights):
        return trapezoid((phase*transform(q,g,s,weights)).real,dx=h)
    return invert(load)/invert(0*load)
