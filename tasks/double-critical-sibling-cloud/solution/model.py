from functools import lru_cache

import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize_scalar
from scipy.special import beta,ndtr

BOUNDS=(.68,1.33)


def band(v):
    return tuple(z if z is not None else inf for z,inf in zip(v,(-np.inf,np.inf)))


def interval(mu,scale,lo,hi):
    if lo>=hi:
        return 0.
    return float(ndtr((mu-lo)/scale)-ndtr((mu-hi)/scale))


@lru_cache(maxsize=1400)
def rectangle(cut,mu,s,X,Y,pooled):
    xa,xb=X
    ya,yb=Y
    xa=max(xa,cut)
    if xb<=xa:
        return 0.
    if not pooled:
        return interval(mu[0],s[0],xa,xb)*interval(mu[1],s[1],max(ya,-cut),yb)
    xa=max(xa,-yb,mu[0]-15*s[0])
    xb=min(xb,mu[0]+15*s[0])
    if xb<=xa:
        return 0.
    def f(x):
        y=interval(mu[1],s[1],max(ya,-x),yb)
        return y*np.exp(-.5*((x-mu[0])/s[0])**2)/(s[0]*np.sqrt(2*np.pi))
    return quad(f,xa,xb,epsabs=2e-10,epsrel=3e-9,points=[-ya] if xa < -ya < xb else None)[0]


def read(e,g,pooled=True):
    v=np.asarray(e["variances"])
    if e["observable"]=="warm_energy":
        return -e["inverse_temperature"]*g*g*v.sum()
    c=np.sqrt(2*e["ratio"])/g
    cut=-e["detuning"]/c
    sig=tuple(g*np.sqrt(v))
    mu=tuple((c*np.array(e["fields"])-e["window"])*np.array(sig)**2)
    full=(-np.inf,np.inf)
    Z=rectangle(cut,mu,sig,full,full,pooled)
    if e["observable"]=="thermal_band":
        return rectangle(cut,mu,sig,band(e["family_band"]),band(e["site_band"]),pooled)/Z
    C=interval(mu[1],sig[1],-cut,np.inf)
    density=np.exp(-.5*((cut-mu[0])/sig[0])**2)/(np.sqrt(2*np.pi)*sig[0])
    n,m=len(e["inside_bands"]),len(e["outside_bands"])
    value=density*C/(c*Z)*beta(n-1,m+1)
    for x in e["inside_bands"]:
        low,high=band(x)
        value*=interval(mu[1],sig[1],max(low,-cut),high)/C
    for x,y in e["outside_bands"]:
        value*=rectangle(cut,mu,sig,band(x),band(y),pooled)/Z
    return float(value)


def _response(experiments,gain):
    return np.array([read(e,gain) for e in experiments])


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
        self.gain = float(result.x)
        return self

    def predict(self, experiments):
        return _response(experiments, self.gain)
