import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.optimize import minimize_scalar

BOUNDS=(.69,1.31)
X,W=leggauss(38)


def box(lo,hi):
    return (hi+lo)/2+(hi-lo)*X/2,(hi-lo)*W/2


def normal(x,v):
    return np.exp(-x*x/(2*v))/np.sqrt(2*np.pi*v)


def answer(e,g,cone=True):
    vf,vl=g*g*np.asarray(e["variances"])
    if e["observable"]=="warm_energy":
        return -e["inverse_temperature"]*(vf+vl)
    b=e["center"]
    c=np.sqrt(2*e["ratio"])/g
    base=[]
    marker=[]
    lo,hi=e["address_band"]
    for setting in e["settings"]:
        m=setting["mixing"]
        a,z=c*np.asarray(setting["fields"])-setting["window"]
        l,h=setting["gate"]
        lo=max(lo,l/(1-m))
        hi=min(hi,h/(1-m))
        if b==0.:
            area=(h*h-l*l)/(2*m*(1-m if cone else 1.))
            base.append(area*normal(0.,vf)*normal(0.,vl))
        else:
            x,wx=box(0.,b)
            y=((1-m)*b-x)/m
            weights=np.exp(a*x+z*y)*normal(x,vf)*normal(y,vl)
            base.append((h-l)*(wx@weights)/m)
        marker.append(np.exp((a-z)*b))
    if hi<=lo:
        return 0.
    t0,t1=e["line_window"]
    w,wt=box(np.exp(-c*t1),np.exp(-c*t0))
    D=np.array(base)[:,None]+np.array(marker)[:,None]*w
    value=(wt@(marker[0]*marker[1]/(D[0]*D[1])))/c
    return float((hi-lo)*normal(b,vf)*normal(b,vl)*value)


def _response(experiments,gain):
    return np.array([answer(e,gain) for e in experiments],dtype=float)


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
