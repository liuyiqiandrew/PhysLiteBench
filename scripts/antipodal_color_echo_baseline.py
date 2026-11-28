import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import ndtr
from numpy.polynomial.legendre import leggauss

BOUNDS=(.72,1.32)
NODES,WEIGHTS=leggauss(38)
X=(NODES+1)/2
W=WEIGHTS/2
SQRT2PI=np.sqrt(2*np.pi)


def answer(e,gain,crossed=True):
    if e["observable"]=="coil":
        return gain*e["current"]*e["turns"]
    vf,vl=e["variances"]
    v=gain**2*(vf+vl)
    c=np.sqrt(2*e["ratio"])/gain
    k=c*(vf+vl)/vl
    sv,sl=np.sqrt(v),gain*np.sqrt(vl)
    pref=1/(2*np.pi*sl**2)
    B=[]
    for j,q in enumerate(e["windows"]):
        high=max(0.,-q*v+13*sv)
        s=high*X
        p=np.exp(-.5*((s+q*v)/sv)**2+.5*q*q*v)/(SQRT2PI*sv)
        value=high*(W@(p*ndtr(s*np.sqrt(vf/vl)/sv)))
        if e["leakage"][j]==0:
            value*=1+np.exp(-c*e["offsets"][j])
        B.append(value)
    outside=e["outside_setting"]
    lo,hi=e["root_band"]
    if e["observable"]=="slab_echo":
        lo=max(lo,-e["leakage"][0]/2)
        hi=min(hi,e["leakage"][1]/2)
        if lo>=hi:
            return 0.
        rf=gain*np.sqrt(vf*vl/(vl+2*vf))
        out=pref*np.sqrt(vl/(vl+2*vf))*(ndtr(hi/rf)-ndtr(lo/rf))
        for j,(start,stop) in enumerate(e["line_windows"]):
            t=start+(stop-start)*X
            p=1/(1+B[j]*np.exp(c*t))
            values=np.exp(k*t)*p
            if outside==j:
                values*=1-p
            out*=float((stop-start)*(W@values))
        return float(out)
    root=lo+(hi-lo)*X
    weights=[]
    factors=[]
    for start,stop in e["line_windows"]:
        lower,upper=np.exp(-c*stop),np.exp(-c*start)
        mass=lower+(upper-lower)*X
        weights.append((upper-lower)*W/(c*mass**(1+k/c)))
        factors.append(mass)
    a=factors[0][None,:,None]
    b=factors[1][None,None,:]
    D0=B[0]+a
    D1=B[1]+b
    if crossed:
        D0=D0+b*np.exp(-c*(2*root+e["offsets"][0]))[:,None,None]
        D1=D1+a*np.exp(-c*(-2*root+e["offsets"][1]))[:,None,None]
    values=a*b/(D0*D1)
    if outside!=-1:
        values*=B[outside]/(D0 if outside==0 else D1)
    kernel=np.sum(W*(values@weights[1]@weights[0]))
    return float(pref*(hi-lo)*kernel/(SQRT2PI*gain*np.sqrt(vf)))


def _response(experiments,gain):
    return np.array([answer(e,gain,crossed=False) for e in experiments],dtype=float)


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
