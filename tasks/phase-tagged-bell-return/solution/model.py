import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.optimize import minimize_scalar
from scipy.spatial.transform import Rotation
from scipy.special import ndtr

BOUNDS = (.69, 1.31)
X, W = leggauss(42)


def grid(lo, hi):
    return (hi+lo)/2+(hi-lo)*X/2, (hi-lo)*W/2


def normal(x, v):
    return np.exp(-x*x/(2*v))/np.sqrt(2*np.pi*v)


def bank(h, vf, vl, lo=-np.inf, hi=np.inf):
    a, b = h
    mf, ml = a*vf, b*vl
    sl = np.sqrt(vl)
    lo, hi = max(lo, ml-12*sl), min(hi, ml+12*sl)
    edges = [lo, 0., hi] if lo < 0 < hi else [lo, hi]
    total = 0.
    for left, right in zip(edges[:-1], edges[1:]):
        y, w = grid(left, right)
        total += w @ (normal(y-ml, vl)*ndtr((mf-np.maximum(0., -y))/np.sqrt(vf)))
    return total*np.exp((a*mf+b*ml)/2)


def answer(e, gain, coherent=True):
    vf, vl = gain**2*np.array(e["variances"])
    if e["observable"] == "warm_energy":
        return -e["inverse_temperature"]*(vf+vl)
    c = np.sqrt(2*e["ratio"])/gain
    p = np.array([e["positive_fraction"],1-e["positive_fraction"]])
    h, scalar, vector, bulk = [], [], [], []
    outside = e["outside"]
    for j, s in enumerate(e["settings"]):
        h.append(c*np.asarray(s["fields"])-s["window"])
        f = np.asarray(s["internal"]) + np.array([1.,-1.])[:,None]*s["disorder"]
        mag = np.linalg.norm(f,axis=1)
        scale = np.full_like(mag,c)
        np.divide(np.sinh(c*mag),mag,out=scale,where=mag>0)
        scalar.append(np.cosh(c*mag))
        vector.append(-scale[:,None]*f)
        mean = 2 * (p @ scalar[-1])
        bulk.append(mean*bank(h[-1],vf,vl))
        if outside is not None and outside["setting"]==j:
            marked = mean*bank(h[-1],vf,vl,*outside["band"])
    vector[1] = Rotation.from_rotvec(2*gain*np.asarray(e["pulse"])).apply(vector[1])
    lo, hi = e["bridge_band"]
    lo = max(0.,lo)
    if lo>=hi:
        return 0.
    x, wx = grid(lo,hi)
    lo, hi = e["line_window"]
    t, wt = grid(np.exp(-c*hi),np.exp(-c*lo))
    shift = np.exp(np.diff(np.asarray(h),axis=1) * (-x))
    phase = np.exp(1j*(e["phase"]+e["phase_slope"]*x))
    result = 0.
    for m in range(2):
        a, b = vector[0][m], vector[1][m]
        numerator = scalar[0][m]*scalar[1][m]-a[2]*b[2]
        if coherent:
            numerator -= np.real(phase*(a[0]-1j*a[1])*(b[0]+1j*b[1]))
        denom = np.asarray(bulk)[:,None,None]+2*np.asarray(scalar)[:,m,None,None]*shift[:,:,None]*t
        event = (numerator*shift[0]*shift[1])[:,None] / (denom[0]*denom[1])
        if outside is not None:
            event *= marked/denom[outside["setting"]]
        result += p[m] * ((wx*normal(x,vf)*normal(-x,vl)) @ (event@wt))
    return result/c


def _response(experiments, gain):
    return np.asarray([answer(e,gain) for e in experiments], dtype=float)


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
