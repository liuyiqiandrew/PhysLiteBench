import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.optimize import minimize_scalar
from scipy.special import ndtr

BOUNDS = (.69,1.31)
X,W = leggauss(34)
REGIONS = ("remote","row","column","corner","tag")


def grid(a,b):
    return (a+b)/2+(b-a)*X/2,(b-a)*W/2


def normal(x,v):
    return np.exp(-x*x/(2*v))/np.sqrt(2*np.pi*v)


def cloud(h,v,region,band=None):
    if region=="tag":
        return 1.
    lo,hi = (-np.inf,np.inf) if band is None else band
    lo = max(0.,lo)
    if hi<=lo:
        return 0.
    m = v*h
    sd = np.sqrt(v)
    factor = np.exp(m*h/2)
    if region=="corner":
        return factor[2]*(ndtr((hi-m[2])/sd[2])-ndtr((lo-m[2])/sd[2]))
    if region=="remote":
        V = v[0]+v[1]
        center = m[0]+m[1]
        b = min(hi,center+12*np.sqrt(V))
        if b<=lo:
            return 0.
        p,w = grid(lo,b)
        conditional = m[0]+v[0]/V*(p-center)
        residual = np.sqrt(v[0]*v[1]/V)
        sides = ndtr((p-conditional)/residual)-ndtr(-conditional/residual)
        return factor.prod()*w@(normal(p-center,V)*sides*ndtr((p+m[2])/sd[2]))
    i = 1 if region=="row" else 0
    b = min(hi,m[i]+12*sd[i])
    if b<=lo:
        return 0.
    p,w = grid(lo,b)
    return factor[i]*factor[2]*w@(normal(p-m[i],v[i])*ndtr((p+m[2])/sd[2]))


def answer(e,g,with_strips=True):
    v = g*g*np.asarray(e["variances"])
    if e["observable"]=="warm_energy":
        return -e["inverse_temperature"]*sum(v)
    c = np.sqrt(2*e["ratio"])/g
    a,w0 = grid(*e["row_window"])
    b,w1 = grid(*e["column_window"])
    t,w2 = grid(*e["site_window"])
    R,C,I = np.exp(-c*a)[:,None,None],np.exp(-c*b)[None,:,None],np.exp(-c*t)[None,None,:]
    terms = dict(zip(REGIONS,[1.,R,C,R*C,R*C*I]))
    response = np.ones((len(a),len(b),len(t)))
    choices = [{"setting":0,"region":"tag","band":None},*e["draws"]]
    for i,setup in enumerate(e["settings"]):
        h = c*np.asarray(setup["fields"])-setup["window"]
        D = sum(terms[r]*cloud(h,v,r) for r in REGIONS if with_strips or r not in ("row","column"))
        for row in choices:
            if row["setting"]==i:
                q = row["region"]
                response *= terms[q]*cloud(h,v,q,row["band"])/D
    return np.prod(normal(0.,v))*np.einsum("i,j,k,ijk->",w0*np.exp(c*a),w1*np.exp(c*b),w2*np.exp(c*t),response)


def _response(experiments,gain):
    return np.asarray([answer(e,gain,with_strips=False) for e in experiments],dtype=float)


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
