import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.optimize import minimize_scalar
from scipy.special import ndtr, roots_genlaguerre

BOUNDS = (.69, 1.31)
X, W = leggauss(36)
QUAD = {r: roots_genlaguerre(78,r-1) for r in (1,2)}


def grid(a, b):
    return (a+b)/2+(b-a)*X/2, (b-a)*W/2


def pdf(x, v):
    return np.exp(-x*x/(2*v))/np.sqrt(2*np.pi*v)


def ordinary(h, vf, vl):
    a, b = h
    m, s = b*vl, np.sqrt(vl)
    edges = [m-12*s,0.,m+12*s]
    total = 0.
    for lo, hi in zip(edges[:-1],edges[1:]):
        y, dy = grid(lo,hi)
        total += dy @ (pdf(y-m,vl)*ndtr((a*vf-np.maximum(0,-y))/np.sqrt(vf)))
    return total * np.exp((a*a*vf+b*b*vl)/2)


def answer(e, gain, full=True):
    vf, vl = gain**2*np.array(e["variances"])
    if e["observable"] == "warm_energy":
        return -e["inverse_temperature"]*(vf+vl)
    c = np.sqrt(2*e["ratio"])/gain
    depth = e["depth"]
    settings = e["settings"]
    g0, g1 = [s["gate"] for s in settings]
    lo = max(g0[0],g1[0],e["line_window"][0])
    hi = min(g0[1],g1[1],e["line_window"][1])
    if hi<=lo:
        return 0.
    h = np.array([c*np.asarray(s["fields"])-s["window"] for s in settings])
    bulk = np.array([ordinary(p,vf,vl) for p in h])
    bias = depth*(h[:,1]-h[:,0])
    z, wz = grid(*e["root_window"])
    t, wt = grid(lo,hi)
    k = e["retention"]*pdf(depth,vl)*np.exp(-c*z)
    count = k*(np.exp(c*g0[1])-np.exp(c*g0[0]))/c
    site = np.exp(bias[:,None]-c*t)
    base = bulk[1]+site[1]
    power = 2 if e["outside"] else 1
    residual = [(g1[0],min(g1[1],g0[0])),(max(g1[0],g0[1]),g1[1])]
    phi = 0.
    nodes, quadrature = QUAD[power]
    for a,b in residual:
        if full and b>a:
            u, wu = grid(a,b)
            energy = np.exp(bias[1]-c*u)
            phi += -np.expm1(-nodes[:,None,None]*energy/base[:,None]) @ (wu*np.exp(c*u))
    if np.ndim(phi):
        moment = np.einsum("s,szt->zt",quadrature,np.exp(-k[None,:,None]*phi[:,None,:]))/base**power
    else:
        moment = np.broadcast_to(1/base**power,(len(z),len(t)))
    if e["outside"]:
        moment = moment*bulk[1]
    flux = pdf(-depth,vf) * np.exp(c*z-count)*k
    return (wz*flux) @ (moment @ (wt*np.exp(c*t)*site[0]*site[1]/(bulk[0]+site[0])))


def _response(experiments, gain):
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
