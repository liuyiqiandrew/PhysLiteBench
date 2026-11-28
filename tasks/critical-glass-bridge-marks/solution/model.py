from collections import defaultdict
from functools import lru_cache

import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize_scalar
from scipy.special import gamma,ndtr

BOUNDS=(.65,1.34)


def eppf(alpha,theta,counts):
    total=sum(counts)
    if total==1:
        return 1.
    m=len(counts)
    return float(np.prod(theta+alpha*np.arange(1,m))*gamma(1+theta)/gamma(total+theta)
                 *np.prod(gamma(np.asarray(counts)-alpha)/gamma(1-alpha)))


@lru_cache(maxsize=2200)
def rectangle(mean,cov,lower,upper):
    if any(x>=y for x,y in zip(lower,upper)):
        return 0.
    std=np.sqrt(cov[0])
    residual=np.sqrt(cov[2]-cov[1]**2/cov[0])
    lo=(lower[0]-mean[0])/std
    hi=(upper[0]-mean[0])/std

    def pdf(z):
        conditional=mean[1]+(cov[1]/std)*z
        return (np.exp(-z*z/2)/np.sqrt(2*np.pi)
                *(ndtr((upper[1]-conditional)/residual)-ndtr((lower[1]-conditional)/residual)))

    return quad(pdf,lo,hi,epsabs=2e-11,epsrel=2e-10)[0]


def response(e,gain):
    v=np.array(e["variances"])
    beta=e["inverse_temperature"]
    if e["observable"]=="energy_density":
        return -beta*gain**2*v.sum()
    critical=np.sqrt(2*e["inner_ratio"])/gain
    a=np.sqrt(2*e["root_entropy"]/v[0])/(gain*beta)
    b=critical/beta
    t=np.cumsum(v[1:])[:2]
    C=gain**2*(np.minimum.outer(t,t)-np.outer(t,t)/v[1:].sum())
    mu=tuple(critical*C@e["field"])
    cov=(C[0,0],C[0,1],C[1,1])
    support=-np.cumsum(e["detuning"])[:2]/critical
    norm=rectangle(mu,cov,tuple(support),(np.inf,np.inf))
    families=defaultdict(dict)
    for item in e["record"]:
        first,inner,mid,last=item["path"]
        label=(inner,mid,last)
        leaves=families[first]
        if any(key[0]==inner and key!=label for key in leaves):
            return 0.
        if label not in leaves:
            leaves[label]=[0,support.copy(),np.full(2,np.inf)]
        entry=leaves[label]
        entry[0]+=1
        lo=[-np.inf if z is None else z for z in item["lower"]]
        hi=[np.inf if z is None else z for z in item["upper"]]
        entry[1]=np.maximum(entry[1],lo)
        entry[2]=np.minimum(entry[2],hi)
    counts=[sum(leaf[0] for leaf in family.values()) for family in families.values()]
    result=eppf(a,0.,counts)
    for leaves in families.values():
        result*=eppf(b,-a,[leaf[0] for leaf in leaves.values()])
        for _,lo,hi in leaves.values():
            result*=rectangle(mu,cov,tuple(lo),tuple(hi))/norm
    return result


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
