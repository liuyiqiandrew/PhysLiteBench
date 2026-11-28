import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.optimize import minimize_scalar
from scipy.special import gamma, ndtr

BOUNDS = (.69, 1.31)
X, W = leggauss(38)


def grid(a,b):
    return (a+b)/2+(b-a)*X/2,(b-a)*W/2


def pdf(x,v):
    return np.exp(-x*x/(2*v))/np.sqrt(2*np.pi*v)


def ordinary(f,l,vf,vl):
    mean = l*vl
    sd = np.sqrt(vl)
    result = 0.
    for a,b in [(mean-12*sd,0.),(0.,mean+12*sd)]:
        y,wy = grid(a,b)
        result += wy @ (pdf(y-mean,vl)*ndtr((f*vf-np.maximum(0.,-y))/np.sqrt(vf)))
    return np.exp((f*f*vf+l*l*vl)/2)*result


def answer(e,gain,self_mass=True):
    vf,vl = gain**2*np.asarray(e["variances"])
    if e["observable"]=="warm_energy":
        return -e["inverse_temperature"]*(vf+vl)
    c = np.sqrt(2*e["ratio"])/gain
    amplification = e["amplification"]
    alpha = 1/amplification
    f,l = c*np.asarray(e["critical_fields"])-e["critical_window"]
    B = ordinary(f,l,vf,vl)
    m = vl*(c*e["cold_field"]-e["cold_window"])
    sd = np.sqrt(vl)
    L = np.exp(m*m/(2*vl))*ndtr(m/sd)
    z,wz = grid(*e["root_window"])
    t,wt = grid(*e["leaf_window"])
    W0 = np.exp(-amplification*c*z)
    wt0 = np.exp(-c*t)
    A = W0[:,None]*(L+wt0 if self_mass else np.full_like(wt0,L))
    stable = L*(pdf(0.,vf)*gamma(1-alpha)/c)**amplification
    scale = A+stable
    linear = A/scale
    tail = (stable/scale)**alpha
    outside = e["outside_band"]
    if outside is not None:
        low,high = np.maximum(outside,0.)
        band = (ndtr((high-m)/sd)-ndtr((low-m)/sd))/ndtr(m/sd)
    moment = np.zeros_like(A)
    for lo,hi in [(0.,.012),(.012,.2),(.2,1.3),(1.3,6.),(6.,41.)]:
        u,wu = grid(lo,hi)
        kernel = np.exp(-linear[:,:,None]*u**amplification-tail[:,:,None]*u)
        if outside is not None:
            kernel *= alpha*band*tail[:,:,None]*u
        moment += (kernel @ (wu*amplification*u**(amplification-1)))/scale
    return pdf(0.,vf)*pdf(0.,vl)/B*((wz*W0) @ (moment @ (wt*wt0)))


def _response(experiments,gain):
    return np.asarray([answer(e,gain) for e in experiments],dtype=float)


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
