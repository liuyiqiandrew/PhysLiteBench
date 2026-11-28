"""Private reference. A positive-gap chord reaches from 0 to e. Use finite wall slits."""
import numpy as np
from scipy.integrate import solve_ivp


def series(t):
    term,total=1.,1.
    for n in range(1,14):
        term *= (4/3+n-1)/(5/3+n-1)*t
        total += term
    return 1-(1-t)*total


sol=solve_ivp(lambda s,g: [(s-1+(s+1)*g[0])/(3*s*(1-s))],
              (.98,1.e-5),[series(.02)],method="DOP853",dense_output=True,
              rtol=2e-13,atol=5e-15)


def chord(z,w,end):
    # Scaled finite-endpoint probabilities; this form avoids an angular cancellation.
    def angular(p):
        norm=abs(p)*abs(end-p)
        co=(end*p.real-abs(p)**2)/norm
        si=end*p.imag/norm
        small=si**2/(2*(1-co)) if co<0 else (1+co)/2
        return small,si
    s = abs((z-w)/(z-w.conjugate()))**2
    g = series(1-s) if s>=.98 else float(sol.sol(s)[0])
    az,sina=angular(z)
    bw,sinb=angular(w)
    joint=az*bw+sina*sinb*g/4
    return np.array([az,joint])


def avoiding(deriv0,derive,length,newlength):
    return (deriv0*derive*(length/newlength)**2)**(5/8)


def finite(z,w,end,walls,step):
    if not walls:
        return chord(z,w,end)
    x,*others=walls
    a=step*min(abs(x),abs(z-x),abs(w-x),*[abs(x-y) for y in others])**2
    def jac(q):
        return (1+2*a/(q-x)**2)**(-.5)
    # Rationalized difference between square roots, anchored at 0.
    def slit(q):
        return q*(q-2*x)/((q-x)/jac(q)-x/jac(0.))
    newend=slit(end)
    weight=avoiding(jac(0.),jac(end),end,newend)*np.prod([jac(y)**2 for y in others])
    return (finite(z,w,end,others,step)-weight*finite(
        slit(z),slit(w),newend,[slit(y) for y in others],step))/a


def bounded(r,z,w,end,walls,step):
    disk=lambda q:q/(1+(q/r)**2)
    jac=lambda q:(1-(q/r)**2)/(1+(q/r)**2)**2
    newend=disk(end)
    weight=avoiding(jac(0),jac(end),end,newend)*np.prod([jac(x)**2 for x in walls])
    return weight*finite(disk(z),disk(w),newend,[disk(x) for x in walls],step)/end**2


def shell(r,z,w,end,walls,step):
    h=.002*r
    def cumulative(radius):
        return bounded(radius,z,w,end,walls,step)
    return (cumulative(r-2*h)-8*cumulative(r-h)+8*cumulative(r+h)-cumulative(r+2*h))/(12*h)


def estimate(r,z,w,walls,end=.012,step=.004):
    # Even endpoint extrapolation and independent positive-contact extrapolation.
    def symmetric(gap):
        return (shell(r,z,w,gap,walls,step)+shell(r,z,w,-gap,walls,step))/2
    coarse=(4*symmetric(end/2)-symmetric(end))/3
    return coarse


def value(q,aspect):
    z,w=[complex(x,aspect*y) for x,y in q["dyes"]]
    r=q["radius"]
    if q["gate"]=="off":
        # Measure a single dye on the finite chord. The mass density of the radial shell is 5/(4R³).
        return estimate(r,w,z,[])[0]*(4*r**3/5)
    vals=[estimate(r,z,w,q["walls"][::-1],step=.004/2**j) for j in range(3)]
    one,both=(vals[0]-6*vals[1]+8*vals[2])/3
    return both/one
