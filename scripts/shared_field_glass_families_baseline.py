from functools import lru_cache

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import gamma, roots_jacobi

BOUNDS = (.60,1.36)
SEPARATE = False


@lru_cache(maxsize=15)
def points(n,m):
    if n==0 or m==0:
        return np.array([float(m==0)]),np.ones(1)
    x,w=roots_jacobi(100,m-1,n-1)
    return (1+x)/2,w/w.sum()


def leaf_partition(sizes,child,parent):
    sizes=list(sizes)
    count=sum(sizes)
    terms=np.arange(1,len(sizes))*child-parent
    return (np.prod(terms)*gamma(1-parent)*np.prod(gamma(np.array(sizes)-child)/gamma(1-child))
            /gamma(count-parent))


def read(e,gain):
    beta=e["inverse_temperature"]
    if e["observable"]=="energy_density":
        return -beta*gain**2*sum(e["variances"])
    a,b=np.sqrt(2*np.array(e["entropies"])/e["variances"])/(beta*gain)
    field=np.array(e["fields"])
    lamps=np.array(e["settings"],dtype=int)
    n,m=np.bincount(lamps,minlength=2)
    if not SEPARATE:
        field=np.full(2,(field@[n,m])/(n+m))
    factors=np.exp(beta*field[:,None]*np.array([-1.,1.]))
    u,w=points(n,m)
    strength=u[:,None]*factors[0]+(1-u[:,None])*factors[1]
    families={}
    for idx,row in enumerate(e["record"]):
        families.setdefault(row[0],[]).append(idx)
    k=len(families)
    exponent=np.mean(strength**a,axis=1)
    product=np.full(len(u),gamma(k)/(a*gamma(n+m)))
    for members in families.values():
        children={}
        signs=set()
        for i in members:
            leaf=e["record"][i][1]
            children[leaf]=children.get(leaf,0)+1
            if e["root_filter"][i]:
                signs.add(e["root_filter"][i])
        if len(signs)>1:
            return 0.
        multiplicity=len(members)
        f0=np.count_nonzero(lamps[members]==0)
        accept=np.array([not signs or v in signs for v in (-1,1)])
        scale=factors[0]**f0*factors[1]**(multiplicity-f0)
        product*=((strength**(a-multiplicity)) @ (scale*accept/2))
        product*=a*gamma(multiplicity-a)/gamma(1-a)*leaf_partition(children.values(),b,a)
    return float(w@(product/exponent**k))


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
