from collections import Counter, defaultdict
from functools import lru_cache

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import gamma, roots_jacobi

BOUNDS = (.67,1.32)


def eppf(a,theta,counts):
    n=np.asarray(counts)
    return (gamma(1+theta)/gamma(n.sum()+theta)*np.prod(theta+a*np.arange(1,len(n)))
            *np.prod(gamma(n-a)/gamma(1-a)))


@lru_cache(maxsize=420)
def beta_rule(a,b):
    x,w = roots_jacobi(35,b-1,a-1)
    return (x+1)/2,w/w.sum()


def response(e,gain):
    var=np.asarray(e["variances"])
    beta=e["inverse_temperature"]
    if e["observable"]=="warm_energy":
        return -beta*gain*gain*var.sum()
    a,b=np.sqrt(2*np.asarray(e["entropies"])/var)/(beta*gain)
    record=[tuple(r) for r in e["record"]]
    edited=record[:-1]
    counts=Counter(record)
    groups=defaultdict(dict)
    for (f,s),n in counts.items():
        groups[f][s]=n
    prob=1.
    params=[]
    def allocate(occupied,base,theta,selected):
        nonlocal prob
        prob*=eppf(base,theta,list(occupied.values()))
        total=sum(occupied.values())+theta
        result={}
        for label in selected:
            weight=occupied[label]-base
            total-=weight
            result[label]=len(params)
            params.append((weight,total))
        return result
    root_counts={f:sum(values.values()) for f,values in groups.items()}
    visited=list(dict.fromkeys(f for f,_ in edited))
    families=allocate(root_counts,a,0,visited)
    sites={}
    for family in groups:
        inside=list(dict.fromkeys(s for f,s in edited if f==family))
        sites[family]=allocate(groups[family],b,-a,inside)
    quadrature=[beta_rule(u,v) for u,v in params]
    nodes=np.ix_(*[z for z,w in quadrature])
    wk=np.ix_(*[w for z,w in quadrature])
    weights=np.prod(np.broadcast_arrays(*wk),axis=0)
    def masses(labels):
        remaining=1.
        values={}
        for label,j in labels.items():
            take=remaining*nodes[j]
            values[label]=take
            remaining=remaining-take
        return values
    P=masses(families)
    Q={f:masses(s) for f,s in sites.items()}
    family_shift=defaultdict(float)
    site_shift=defaultdict(float)
    contribution=1.
    for i,((f,s),(df,ds)) in enumerate(zip(edited,e["edits"])):
        family_shift[f]+=beta*df
        site_shift[f,s]+=beta*ds
        z=1.
        for k,h in family_shift.items():
            factor=np.exp(h)
            z=z+P[k]*np.expm1(h)
            for (ff,ss),t in site_shift.items():
                if ff==k:
                    z=z+factor*P[k]*Q[k][ss]*np.expm1(t)
        f1,s1=record[i+1]
        numerator=np.exp(family_shift.get(f1,0.)+site_shift.get((f1,s1),0.))
        contribution=contribution*numerator/z
    return float(prob*np.sum(weights*contribution))


def _response(experiments,gain):
    return np.array([response(e,gain) for e in experiments])


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
