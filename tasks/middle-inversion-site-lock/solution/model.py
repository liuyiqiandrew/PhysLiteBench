import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.optimize import minimize_scalar
from scipy.special import ndtr

BOUNDS=(.68,1.31)
X,W=leggauss(48)
Q,P=leggauss(34)
S2=np.sqrt(2*np.pi)


def rule(lo,hi,nodes=X,weights=W):
    return (lo+hi)/2+(hi-lo)*nodes/2,(hi-lo)*weights/2


def normal(x,v):
    return np.exp(-x*x/(2*v))/(S2*np.sqrt(v))


def masses(v,q):
    center=-q*v
    limit=np.maximum(0.,np.cumsum(center)+12*np.sqrt(np.cumsum(v)))
    root,w0=rule(0.,limit[0])
    prefix,w1=rule(0.,limit[1])
    inner=normal(prefix[None,:]-root[:,None]-center[1],v[1])
    p2=ndtr((prefix+center[2])/np.sqrt(v[2]))
    all_mass=(w0*normal(root-center[0],v[0]))@(inner@(w1*p2))
    all_mass*=np.exp(.5*q*q*sum(v))
    x,wx=rule(0.,max(0.,center[1]+12*np.sqrt(v[1])))
    sub=np.exp(.5*q*q*sum(v[1:]))*(wx@(normal(x-center[1],v[1])
                                       *ndtr((x+center[2])/np.sqrt(v[2]))))
    return all_mass,sub


def answer(e,g,include_subtree=True):
    v=g*g*np.asarray(e["variances"])
    if e["observable"]=="warm_energy":
        return -e["inverse_temperature"]*sum(v)
    c=np.sqrt(2*e["ratio"])/g
    k=c*(v[1]+v[2])/v[2]
    B,L=np.array([masses(v,q) for q in e["temperature_windows"]]).T
    t,wt=rule(*e["root_window"],Q,P)
    root=np.exp(-c*t)
    root_mass=L[:,None]*root[None,:]
    points,weights=[],[]
    for lo,hi in e["line_windows"]:
        p,w=rule(np.exp(-c*hi),np.exp(-c*lo),Q,P)
        points.append(p)
        weights.append(w*p**(-k/(2*c))/c)
    d0=B[0]+points[0][None,:,None]+include_subtree*root_mass[0,:,None,None]
    d1=B[1]+points[1][None,None,:]+include_subtree*root_mass[1,:,None,None]
    event=1/(d0*d1)
    o=e["additional"]
    if o is not None:
        j=o["setting"]
        mass=B[j] if o["region"]=="different_root" else root_mass[j,:,None,None]
        event*=mass/(d0 if j==0 else d1)
    measure=wt*np.exp((c-k)*t)
    return float(measure@(event@weights[1]@weights[0])/(2*S2**3*np.sqrt(np.prod(v))))


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
