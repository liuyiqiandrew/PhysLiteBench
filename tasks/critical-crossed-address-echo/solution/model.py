from functools import lru_cache

import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize_scalar
from scipy.special import ndtr,roots_legendre

BOUNDS=(.68,1.33)
_N,_W=roots_legendre(35)
FULL=(-np.inf,np.inf)


def bands(z):
    return tuple(inf if x is None else x for x,inf in zip(z,FULL))


def prob(mu,s,lo,hi):
    return 0. if lo>=hi else float(ndtr((mu-lo)/s)-ndtr((mu-hi)/s))


def pdf(x,m,s):
    return np.exp(-.5*((x-m)/s)**2)/(np.sqrt(2*np.pi)*s)


def strip(x,cut,mu,s,Y,Z):
    ya,yb=Y
    za,zb=Z
    lo=max(ya,cut,-x-zb,mu[0]-14*s[0])
    hi=min(yb,mu[0]+14*s[0])
    if lo>=hi:
        return 0.
    def density(y):
        return pdf(y,mu[0],s[0])*prob(mu[1],s[1],max(za,-x-y),zb)
    v=-x-za
    return quad(density,lo,hi,epsabs=2e-10,epsrel=3e-9,
                points=[v] if lo<v<hi else None)[0]


@lru_cache(maxsize=700)
def cloud(a,b,mu,s,X,Y,Z):
    lo=max(X[0],a,mu[0]-13*s[0],-Y[1]-Z[1])
    hi=min(X[1],mu[0]+13*s[0])
    if lo>=hi:
        return 0.
    points=set(-y-z for y in (max(b,Y[0]),Y[1]) for z in Z)
    nodes=sorted(p for p in points if lo<p<hi and np.isfinite(p))
    return quad(lambda x:pdf(x,mu[0],s[0])*strip(x,b,mu[1:],s[1:],Y,Z),
                lo,hi,points=nodes,epsabs=3e-10,epsrel=5e-9)[0]


def displacement(window,c,scale):
    lo,hi=window
    h=(hi-lo)/2
    x=(hi+lo)/2+h*_N
    t=np.exp(c*x)
    return scale/t,h*_W*t


def read(e,g,strips=True):
    V=np.asarray(e["variances"])
    if e["observable"]=="warm_energy":
        return -g*g*e["inverse_temperature"]*V.sum()
    c=np.sqrt(2*e["ratio"])/g
    a,b=-np.asarray(e["detunings"])/c
    sig=tuple(g*np.sqrt(V))
    h=c*np.asarray(e["fields"])-e["window"]
    mu=tuple(h*np.asarray(sig)**2)
    Q=cloud(a,b,mu,sig,FULL,FULL,FULL)
    if e["observable"]=="thermal_gate":
        return cloud(a,b,mu,sig,*map(bands,e["bands"]))/Q
    U=strip(a,b,mu[1:],sig[1:],FULL,FULL) if strips else 0.
    V=strip(b,a,(mu[0],mu[2]),(sig[0],sig[2]),FULL,FULL) if strips else 0.
    W=prob(mu[2],sig[2],-a-b,np.inf)
    p,pw=displacement(e["row_window"],c,np.exp(h[0]*a-.5*(h[0]*sig[0])**2))
    q,qw=displacement(e["column_window"],c,np.exp(h[1]*b-.5*(h[1]*sig[1])**2))
    both=W*p[:,None]*q
    normalization=Q+p[:,None]*U+q*V+both
    volume=pw @ ((both/normalization)**len(e["internal_bands"])) @ qw
    selection=1.
    for gate in e["internal_bands"]:
        lo,hi=bands(gate)
        selection*=prob(mu[2],sig[2],max(lo,-a-b),hi)/W
    return float(pdf(a,0.,sig[0])*pdf(b,0.,sig[1])*volume*selection)


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
