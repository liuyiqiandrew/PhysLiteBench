from collections import defaultdict
from itertools import product
from functools import lru_cache

import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize_scalar
from scipy.special import gamma

BOUNDS=(.65,1.35)


@lru_cache(maxsize=1024)
def radial(a,K,power):
    def integrand(log_t):
        if log_t>300:
            return 0.
        return np.exp(power*log_t-np.dot(K,np.exp(np.asarray(a)*log_t)))
    return quad(integrand,-np.inf,np.inf,epsabs=2e-11,epsrel=1e-9,limit=220)[0]


def probability(e,gain):
    beta=e["inverse_temperature"]
    v=np.asarray(e["variances"])
    if e["observable"]=="bare_energy_density":
        return -beta*gain**2*v[e["phase"]].sum()
    s=np.asarray(e["entropies"])
    a,b=(np.sqrt(2*s/v)/(beta*gain)).T
    q=a/b
    c=1/np.sqrt(4*np.pi*s)
    leaf_scale=c[:,1]*gamma(1-b)
    K=c[:,0]*gamma(1-q)*leaf_scale**q*np.exp(-beta*a*np.asarray(e["offset"]))
    log_bias=np.asarray(e["log_bias"])
    available=np.flatnonzero(log_bias==log_bias.min())
    for phase in (0,1):
        if phase not in available:
            K[phase]=0.
    groups=defaultdict(dict)
    filters=defaultdict(set)
    for family,leaf,phase in e["record"]:
        group=groups[family]
        group[leaf]=group.get(leaf,0)+1
        if phase!=-1:
            filters[family].add(phase)
    multipliers=[]
    for f,leaves in groups.items():
        req=filters[f]
        sizes=np.asarray(list(leaves.values()))
        m=len(sizes)
        row=[]
        for k in available:
            if req and req!={k}:
                continue
            children=np.prod(b[k]*gamma(sizes-b[k])/gamma(1-b[k]))
            factor=K[k]*q[k]*gamma(m-q[k])/gamma(1-q[k])*children
            row.append((k,float(factor)))
        if not row:
            return 0.
        multipliers.append(row)
    total=0.
    for assignment in product(*multipliers):
        phase,coef=zip(*assignment)
        power=float(a[list(phase)].sum())
        total+=np.prod(coef)*radial(tuple(a),tuple(K),power)
    return float(total/gamma(len(e["record"])))


def _response(experiments,gain):
    return np.array([probability(e,gain) for e in experiments])


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
