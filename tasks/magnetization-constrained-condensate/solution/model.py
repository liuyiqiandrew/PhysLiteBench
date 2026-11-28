from functools import lru_cache
from math import factorial,gamma

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq,minimize_scalar
from scipy.special import zeta

BOUNDS=(.69,1.30)
J=np.array([-1,0,1])
SERIES=np.array([zeta(1.5-k)*(-1)**k/factorial(k) for k in range(13)])
SERIES_INDEX=np.arange(1.,71.)


def occupation(gap):
    gap=max(gap,0.)
    if gap<.46:
        return np.polynomial.polynomial.polyval(gap,SERIES)+gamma(-.5)*np.sqrt(gap)
    return np.sum(np.exp(-gap*SERIES_INDEX)/SERIES_INDEX**1.5)


@lru_cache(maxsize=240)
def condensate(density,magnetization,temperature,quadratic,mass):
    levels=quadratic*J**2/temperature
    scale=(mass*temperature/(2*np.pi))**1.5

    def excited(eta):
        shifted=levels-eta*J
        return scale*np.array([occupation(x-shifted.min()) for x in shifted])

    # Corners of the feasible region may support two or three condensates.
    contacts=sorted(set((levels[b]-levels[a])/(J[b]-J[a]) for a,b in [(0,1),(0,2),(1,2)]))
    for eta in contacts:
        shifted=levels-eta*J
        active=np.flatnonzero(abs(shifted-shifted.min())<1e-12)
        if len(active)<2:
            continue
        thermal=excited(eta)
        total=density-thermal.sum()
        spin=magnetization-thermal@J
        if total<=0 or not J[active[0]]*total<=spin<=J[active[-1]]*total:
            continue
        if len(active)==3:
            plus=max(spin,0.)
            minus=max(-spin,0.)
            return (minus,total-abs(spin),plus), (total-abs(spin))/2
        a,b=active
        c=np.zeros(3)
        c[b]=(spin-J[a]*total)/(J[b]-J[a])
        c[a]=total-c[b]
        return tuple(c),0.

    # On each open face only the spin of the lowest line can condense.
    cuts=[-60.,*contacts,60.]
    for left,right in zip(cuts,cuts[1:]):
        index=int(np.argmin(levels-(left+right)*J/2))
        delta=J-J[index]
        target=magnetization-J[index]*density
        def imbalance(eta):
            return excited(eta)@delta-target
        if imbalance(left)>0 or imbalance(right)<0:
            continue
        eta=brentq(imbalance,left,right,xtol=2e-13)
        total=density-excited(eta).sum()
        if total>0:
            c=np.zeros(3)
            c[index]=total
            return tuple(c),0.
    return (0.,0.,0.),0.


def _predict(e,mass):
    rho,T=e["density"],e["temperature"]
    c,span=condensate(rho,e["magnetization"],T,e["quadratic"],mass)
    observable=e["observable"]
    if observable=="zero_fraction":
        return sum(c)/rho
    lo,hi=np.array(e["window"])*rho
    center=(c[2]-c[0])*np.cos(2*e["pulse_angle"])
    ramp=2*abs(np.sin(2*e["pulse_angle"]))
    def gate(v):
        amplitude=ramp*np.sqrt((c[2]+v)*(c[0]+v))
        if amplitude<1e-14:
            return float(lo<=center<=hi)
        left,right=np.clip([(lo-center)/amplitude,(hi-center)/amplitude],-1,1)
        return (np.arcsin(right)-np.arcsin(left))/np.pi

    if span<=0:
        return gate(0.)*(c[1]/rho if observable=="neutral_window_mean" else 1.)
    slope=-np.dot(e["split"],[1,-2,1])*span/T
    def integrand(x):
        v=span*x
        value=np.exp(slope*x)*gate(v)
        return value*(c[1]-2*v)/rho if observable=="neutral_window_mean" else value
    breaks=[]
    if ramp:
        for endpoint in [lo,hi]:
            root=(np.sqrt((c[2]-c[0])**2+4*((endpoint-center)/ramp)**2)-c[0]-c[2])/2/span
            if 0<root<1:
                breaks.append(root)
    return quad(integrand,0,1,points=breaks,epsabs=3e-8)[0]/quad(lambda x:np.exp(slope*x),0,1)[0]


def _response(experiments,mass):
    return np.array([_predict(e,mass) for e in experiments])


class Model:
    def fit(self, records):
        inputs = [r["input"] for r in records]
        measured = np.array([r["value"] for r in records])
        sigma = np.array([r["sigma"] for r in records])
        def chi2(parameter):
            residual = (_response(inputs, parameter) - measured) / sigma
            return residual @ residual
        result = minimize_scalar(chi2, bounds=BOUNDS, method="bounded",
                                 options={"xatol": 1e-10})
        if not result.success:
            raise RuntimeError(result.message)
        self.mass = float(result.x)
        return self

    def predict(self, experiments):
        return _response(experiments, self.mass)
