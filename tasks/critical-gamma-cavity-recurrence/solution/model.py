import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import ndtr, roots_legendre

BOUNDS=(.68,1.33)
X,W=roots_legendre(48)
X,W=(1+X)/2,W/2
S2P=np.sqrt(2*np.pi)


def limits(b):
    lo,hi=b
    return (-np.inf if lo is None else lo,np.inf if hi is None else hi)


def normal(x,m,s):
    return np.exp(-.5*((x-m)/s)**2)/(S2P*s)


def mass(e,c,v,band,rectangle):
    u=np.array([1.,-1.])
    precision=np.diag(1/v)+2*c*e["curvature"]*np.outer(u,u)
    covariance=np.linalg.inv(precision)
    tilt=c*e["field"]*u-e["window"]
    mean=covariance@tilt
    transform=np.array([[1.,1.],[1.,-1.]])
    shifted=transform@mean
    cov=transform@covariance@transform.T
    sd=np.sqrt(cov[0,0])
    end=shifted[0]+13*sd
    m=shifted[1]
    k=cov[0,1]/cov[0,0]
    d=np.sqrt(cov[1,1]-k*cov[0,1])
    left,right=limits(band)
    start=max(0.,left if rectangle else 0.,-right)
    if end<=start:
        return 0.
    cuts=sorted(set([start,end]+[cut for cut in (-left,right) if start<cut<end]))
    total=0.
    for lo,hi in zip(cuts,cuts[1:]):
        q=lo+(hi-lo)*X
        center=m+k*(q-shifted[0])
        low=np.maximum(left,-q)
        high=np.minimum(right,q) if rectangle else right
        probability=ndtr((high-center)/d)-ndtr((low-center)/d)
        total+=(hi-lo)*(W @ (normal(q,shifted[0],sd)*probability))
    return total*np.exp(.5*(mean@tilt))/np.sqrt(v.prod()*np.linalg.det(precision))


def answer(e,g,rectangle=False):
    if e["observable"]=="warm_energy":
        return g*sum(e["shapes"])/(1+g*e["inverse_temperature"])
    c=e["tilt"]/g
    v=np.asarray(e["shapes"])*(g/(1+e["tilt"]))**2
    B=mass(e,c,v,[None,None],rectangle)
    band=e["contrast_band"]
    if e["observable"]=="contrast_gate":
        return mass(e,c,v,band,rectangle)/B
    lo,hi=limits(band)
    r=np.sqrt(v.prod()/v.sum())
    lo=max(0.,lo/2)
    hi=min(hi/2,13*r)
    if hi<=lo:
        return 0.
    x=lo+(hi-lo)*X
    a=np.exp(c*(e["field"]*2*x-e["curvature"]*4*x*x))
    p,q=e["energy_window"]
    p,q=np.exp(-c*q),np.exp(-c*p)
    odds=a*a/((B+p*a)*(B+q*a))
    return float((hi-lo)*(q-p)*(W @ (normal(x,0.,r)*odds))/(c*S2P*np.sqrt(v.sum())))


def _response(experiments,gain):
    return np.array([answer(e,gain) for e in experiments])


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
