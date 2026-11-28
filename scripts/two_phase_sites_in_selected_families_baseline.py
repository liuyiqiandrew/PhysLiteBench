from collections import defaultdict
from functools import lru_cache

import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize_scalar
from scipy.special import gamma

BOUNDS=(.66,1.33)


@lru_cache(maxsize=1300)
def radial(K,b,p):
    K=np.asarray(K)
    b=np.asarray(b)
    def integrand(x):
        if x>190:
            return 0.
        return np.exp(p*x-K@np.exp(b*x))
    return quad(integrand,-np.inf,np.inf,epsabs=2e-11,epsrel=2e-10,limit=200)[0]


@lru_cache(maxsize=150)
def fractional(K,b,a):
    if not a:
        return 1.
    K=np.asarray(K)
    b=np.asarray(b)
    def integrand(x):
        ψ=K@np.exp(b*x)
        if ψ<1e-6:
            factor=1-ψ/2+ψ*ψ/6
            return float((K@np.exp((b-a)*x))*factor)
        return float(np.exp(-a*x)*(-np.expm1(-ψ)))
    L=quad(integrand,-np.inf,0.,epsabs=3e-11,epsrel=2e-9,limit=200)[0]
    R=quad(integrand,0.,32.,epsabs=3e-11,epsrel=2e-9,limit=200)[0]
    return float((L+R+np.exp(-32*a)/a)*a/gamma(1-a))


def read(e,g):
    beta=e["inverse_temperature"]
    v=np.asarray(e["variances"])
    if e["observable"]=="bare_energy_density":
        return -beta*g*g*(v[0]+v[e["phase"]+1])
    s=np.asarray(e["entropies"])
    indexes=np.sqrt(2*s/v)/(g*beta)
    a=0.
    b=indexes[1:]
    K=np.exp(-beta*b*np.asarray(e["offsets"]))/np.sqrt(4*np.pi*s[1:])*gamma(1-b)
    records=defaultdict(list)
    for site,phase in e["record"]:
        records[site].append(phase)
    count=len(e["record"])
    terms={0:1.}
    for children in records.values():
        allowed=[r for r in range(2) if all(q in (-1,r) for q in children)]
        coefficients=b*K*gamma(len(children)-b)/gamma(1-b)
        next_terms=defaultdict(float)
        for used,f in terms.items():
            for r in allowed:
                next_terms[used+r]+=f*coefficients[r]
        terms=next_terms
    total=0.
    m=len(records)
    for assigned,coefficient in terms.items():
        p=(m-assigned)*b[0]+assigned*b[1]-a
        total+=coefficient*radial(tuple(K),tuple(b),float(p))
    scale=gamma(count-a)*fractional(tuple(K),tuple(b),float(a))
    return float(total/scale)


def _response(experiments,g):
    return np.array([read(e,g) for e in experiments])


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
