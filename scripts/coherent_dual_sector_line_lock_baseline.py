import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.optimize import minimize_scalar
from scipy.special import ndtr

BOUNDS = (.69,1.31)
X,W = leggauss(36)


def grid(a,b):
    return a+(b-a)*(X+1)/2,W*(b-a)/2


def gaussian(x,v):
    return np.exp(-x*x/(2*v))/np.sqrt(2*np.pi*v)


def ordinary(h,vf,vl,lo=-np.inf,hi=np.inf):
    f,l = h
    mu,sd = l*vl,np.sqrt(vl)
    a,b = max(lo,mu-12*sd),min(hi,mu+12*sd)
    cuts = [a,0.,b] if a<0.<b else [a,b]
    total = 0.
    for left,right in zip(cuts[:-1],cuts[1:]):
        y,wy = grid(left,right)
        total += wy@(gaussian(y-mu,vl)*ndtr((f*vf-np.maximum(0.,-y))/np.sqrt(vf)))
    return total*np.exp((f*f*vf+l*l*vl)/2)


def answer(e,gain,coherent=True):
    vf,vl = gain**2*np.asarray(e["variances"])
    if e["observable"]=="warm_energy":
        return -e["inverse_temperature"]*(vf+vl)
    c = np.sqrt(2*e["ratio"])/gain
    t0,w0 = grid(*e["line_windows"][0])
    t1,w1 = grid(*e["line_windows"][1])
    z0,z1 = e["root_window"]
    q,wz = grid(np.exp(-c*z1),np.exp(-c*z0))
    result = np.ones((len(q),len(t0),len(t1)))
    outside = e["outside"]
    for j,s in enumerate(e["settings"]):
        h = c*np.asarray(s["fields"])-s["window"]
        colors = np.exp(-c*np.asarray(s["offsets"]))
        B = colors.sum()*ordinary(h,vf,vl)
        l = h[1]*np.sqrt(vl)
        L = colors.sum()*np.exp(l*l/2)*ndtr(l)
        a,b = t0[:,None]+s["offsets"][0],t1+s["offsets"][1]
        bias = (a-b)/2
        mix = s["coupling"] if coherent else 0.
        mag = np.hypot(bias,mix)
        slope = np.full_like(mag,c)
        np.divide(np.sinh(c*mag),mag,out=slope,where=mag>0.)
        heat = np.cosh(c*mag)
        center = np.exp(-c*(a+b)/2)
        P = bias*np.cos(2*s["angle"])+mix*np.sin(2*s["angle"])*np.cos(s["phase"])
        full = B+q[:,None,None]*L+2*center*heat
        result *= center*(heat-slope*P)/full
        if outside is not None and outside["setting"]==j:
            remote = colors[outside["sector"]]*ordinary(h,vf,vl,*outside["band"])
            result *= remote/full
    weights = [wz,w0*np.exp(c*t0),w1*np.exp(c*t1)]
    return gaussian(0.,vf)*gaussian(0.,vl)**2*np.einsum("z,i,j,zij->",*weights,result)/c


def _response(experiments,gain):
    return np.asarray([answer(e,gain,coherent=False) for e in experiments],dtype=float)


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
