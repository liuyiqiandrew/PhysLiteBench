from itertools import product
from math import factorial

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.optimize import minimize_scalar
from scipy.special import ndtr

BOUNDS=(.68,1.31)
X,W=leggauss(36)
MASK=np.array([[1,0],[0,1],[1,1]])
SQ2PI=np.sqrt(2*np.pi)


def pmf(rate,count):
    left,right=count
    if min(left,right)<0:
        return np.zeros_like(rate[0])
    total=0.
    for n in range(min(left,right)+1):
        factors=np.power(rate,np.array([left-n,right-n,n])[:,None])
        total+=factors.prod(axis=0)/(factorial(left-n)*factorial(right-n)*factorial(n))
    return np.exp(-rate.sum(axis=0))*total


def answer(e,gain,separate=False):
    vf,vl=gain*gain*np.array(e["variances"])
    if e["observable"]=="warm_energy":
        return -e["inverse_temperature"]*(vf+vl)
    br=np.sqrt(vf*vl/(vf+vl))
    total=np.sqrt(vf+vl)
    c=np.sqrt(2*e["ratio"])/gain
    fields=e["fields"]
    shells=e["shells"]

    def edges(requirements):
        slopes,lo,hi=[],[],[]
        for j,band in requirements:
            slopes.append(fields[j])
            lo.append(band[0])
            hi.append(band[1])
        return np.array(slopes),np.array(lo),np.array(hi)

    def mass(requirements,bounds,tag):
        a,lower,upper=edges(requirements)
        if tag:
            low,high=max(lower),min(upper)
            return (np.exp(c*high)-np.exp(c*low))/(c*SQ2PI*np.sqrt(vl)) if low<high else 0.
        lo,hi=0.,np.inf
        if bounds is not None:
            lo,hi=bounds
            lo=max(0.,lo)
        if lo>=hi:
            return 0.
        cuts=[lo,hi]
        lines=list(zip(np.tile(a,2),np.concatenate((lower,upper))))
        for i,(m,b) in enumerate(lines):
            for n,d in lines[:i]:
                if m!=n:
                    x=(d-b)/(m-n)
                    if lo<x<hi:
                        cuts.append(x)
        cuts=np.unique(cuts)
        value=0.
        for start,stop in zip(cuts[:-1],cuts[1:]):
            mid=(start+stop)/2 if np.isfinite(stop) else start+max(1.,br)
            i=np.argmax(lower+a*mid)
            j=np.argmin(upper+a*mid)
            if lower[i]+a[i]*mid>=upper[j]+a[j]*mid:
                continue
            def tilted(k,b):
                t=c*a[k]*br
                return np.exp(c*b+t*t/2)*(ndtr(stop/br-t)-ndtr(start/br-t))
            value+=max(0.,tilted(j,upper[j])-tilted(i,lower[i]))
        return value/(c*SQ2PI*total)

    def classes(extra=(),bounds=None,tag=False):
        first,second=(0,shells[0]),(1,shells[1])
        a=mass([first,*extra],bounds,tag)
        b=mass([second,*extra],bounds,tag)
        common=mass([first,second,*extra],bounds,tag)
        return np.maximum(0.,np.array([a-common,b-common,common]))

    outside_base=classes()
    count=np.array(e["counts"])
    if separate:
        base=np.array([outside_base[0]+outside_base[2],outside_base[1]+outside_base[2],0.])
    else:
        base=outside_base
    herald=float(pmf(base[:,None],count)[0])
    if e["observable"]=="herald_rate":
        return herald
    requirements=[[(j,shells[j]),(j,e["marks"][j])] for j in (0,1)]
    if e["same_site"]:
        req=[requirements[0]+requirements[1]]
    else:
        req=requirements
    tag=classes(tag=True)
    marked=[classes(x,tag=True) for x in req]
    order=len(marked)
    other=e["outside"]
    divisor=np.prod(count)
    if other is not None:
        j=other["setting"]
        marked.append(classes([(j,shells[j]),(j,other["mark"])],other["bridge"]))
        divisor*=count[j]
    lo,hi=e["parent_window"]
    t=(lo+hi)/2+(hi-lo)*X/2
    q=np.exp(-c*t)
    rate=outside_base[:,None]+tag[:,None]*q
    out=0.
    for slots in product(range(3),repeat=len(marked)):
        factors=[row[k] for row,k in zip(marked,slots)]
        if not all(factors):
            continue
        left=count-MASK[list(slots)].sum(axis=0)
        out+=np.prod(factors)*(W@(q**(order-1)*pmf(rate,left)))
    return float(out*(hi-lo)/(2*SQ2PI*np.sqrt(vf)*herald*divisor))


def _response(experiments,gain):
    return np.array([answer(e,gain,separate=True) for e in experiments],dtype=float)


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
