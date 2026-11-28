from functools import lru_cache

import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize_scalar
from scipy.special import expit,ndtr

BOUNDS=(.69,1.31)
ALL=(-np.inf,np.inf)


def limits(b):
    return tuple(inf if v is None else v for inf,v in zip(ALL,b))


def prob(m,s,a,b):
    return 0. if a>=b else ndtr((m-a)/s)-ndtr((m-b)/s)


@lru_cache(maxsize=700)
def cloud(mu,sd,X,Y):
    a,b=max(0.,X[0],-Y[1]),min(X[1],mu[0]+14*sd[0])
    if a>=b:
        return 0.
    def f(x):
        return (np.exp(-.5*((x-mu[0])/sd[0])**2)/(sd[0]*np.sqrt(2*np.pi))
                *prob(mu[1],sd[1],max(Y[0],-x),Y[1]))
    return quad(f,a,b,epsabs=2e-11,epsrel=2e-9,points=[-Y[0]] if a < -Y[0] < b else None)[0]


def read(e,g,discrete=True):
    if e["observable"]=="domain_occupation":
        p=e["prior"]
        return expit(np.log(p/(1-p))-e["inverse_temperature"]*g)
    rho=np.asarray(e["fractions"])/20
    sigma=tuple(np.sqrt(rho*(1-rho)))
    t=e["tilt"]
    delta=t/g*np.asarray(e["fields"])-e["window"]
    mu=tuple(delta*np.asarray(sigma)**2)
    C=cloud(mu,sigma,ALL,ALL)
    if e["observable"]=="coordinate_band":
        return cloud(mu,sigma,limits(e["family_band"]),limits(e["site_band"]))/C
    root_density=1/(np.sqrt(2*np.pi)*sigma[0])
    B=prob(mu[1],sigma[1],0.,np.inf)
    ratio=B*np.exp(-.5*(delta[0]*sigma[0])**2)/C
    n,m=len(e["sibling_bands"]),len(e["other_bands"])
    def kernel(z):
        w=np.exp(-t*z)*ratio
        return np.exp(t*z)*(w/(1+w))**n/(1+w)**m
    if discrete:
        weighted=sum(kernel(z) for z in e["addressed_levels"])
    else:
        weighted=sum(quad(kernel,z-.5,z+.5,epsabs=2e-10)[0] for z in e["addressed_levels"])
    out=root_density*weighted
    for row in e["sibling_bands"]:
        lo,hi=limits(row)
        out*=prob(mu[1],sigma[1],max(0.,lo),hi)/B
    for X,Y in e["other_bands"]:
        out*=cloud(mu,sigma,limits(X),limits(Y))/C
    return float(out)


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
