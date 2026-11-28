import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import ndtr, roots_legendre

BOUNDS=(.68,1.33)
NODES,W=roots_legendre(53)
X=(1+NODES)/2
W=W/2
SQRT2PI=np.sqrt(2*np.pi)


def normal(x,m,s):
    return np.exp(-.5*((x-m)/s)**2)/(s*SQRT2PI)


def interval(a):
    return (-np.inf if a[0] is None else a[0],np.inf if a[1] is None else a[1])


def ordinary(s,m,d):
    lo=-d[0]
    hi=max(lo,m[0]+13*s[0])
    x=lo+(hi-lo)*X
    left=(-d[0]-d[1]-x)
    width=np.maximum(0.,m[1]+13*s[1]-left)
    y=left[:,None]+width[:,None]*X
    cutoff=-sum(d)-x[:,None]-y
    inner=(normal(y,m[1],s[1])*ndtr((m[2]-cutoff)/s[2]))@W
    return (hi-lo)*(W @ (width*normal(x,m[0],s[0])*inner))


def answer(e,g,physical_prefix=False):
    v=np.array(e["variances"])*g*g
    if e["observable"]=="warm_energy":
        return -e["inverse_temperature"]*v.sum()
    c=np.sqrt(2*e["ratio"])/g
    d=np.array(e["entropy_shifts"])/c
    a=c*np.array(e["fields"])-e["window"]
    s=np.sqrt(v)
    m=a*v
    background=ordinary(s,m,d)*np.exp(.5*(v[:2]@a[:2]**2))
    leaf=ndtr((m[2]+d[2])/s[2])
    tagged=[]
    for b in e["leaf_bands"]:
        lo,hi=interval(b)
        lo=max(lo,-d[2])
        tagged.append(max(0.,ndtr((m[2]-lo)/s[2])-ndtr((m[2]-hi)/s[2])))

    left,right=interval(e["first_band"])
    total=-d[0]-d[1]
    bridge_mean=total*v[0]/(v[0]+v[1])
    bridge_sd=np.sqrt(v[0]*v[1]/(v[0]+v[1]))
    lo=max(left,bridge_mean-13*bridge_sd,-d[0] if physical_prefix else -np.inf)
    hi=min(right,bridge_mean+13*bridge_sd)
    if lo>=hi or not all(tagged):
        return 0.
    x=lo+(hi-lo)*X
    y=total-x
    mass=np.exp(a[0]*x+a[1]*y)
    p,q=e["prefix_window"]
    low,high=np.exp(-c*q),np.exp(-c*p)
    f=mass*leaf
    denom=(background+low*f)*(background+high*f)
    difference=(high-low)*background*f/denom
    if e["outside"]:
        difference*=.5*background*(1/(background+low*f)+1/(background+high*f))
    weighted=normal(x,0.,s[0])*normal(y,0.,s[1])*mass*difference
    return float((hi-lo)*tagged[0]*tagged[1]*(W@weighted)/(c*leaf*background))


def _response(experiments,gain):
    return np.array([answer(e,gain) for e in experiments])


class Model:
    def fit(self,records):
        inputs=[r["input"] for r in records]
        y=np.array([r["value"] for r in records])
        noise=np.array([r["sigma"] for r in records])

        def chi2(g):
            residual=(_response(inputs,g)-y)/noise
            return residual@residual

        result=minimize_scalar(chi2,bounds=BOUNDS,method="bounded",options={"xatol":1e-10})
        if not result.success:
            raise RuntimeError(result.message)
        self.gain=float(result.x)
        return self

    def predict(self,experiments):
        return _response(experiments,self.gain)


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
