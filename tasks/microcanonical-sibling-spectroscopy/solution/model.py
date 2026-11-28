import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize_scalar
from scipy.special import gamma,gammainc,ndtr

BOUNDS=(.68,1.33)


def bound(band):
    lo,hi=band
    return (-np.inf if lo is None else lo,np.inf if hi is None else hi)


def shell_mass(c,shell,interval):
    lo,hi=bound(interval)
    a,b=max(lo,shell[0]),min(hi,shell[1])
    return 0. if a>=b else (np.exp(c*a)*np.expm1(c*(b-a))/c)


def eval_record(e,g,realized=True):
    var=np.asarray(e["variances"])
    if e["observable"]=="warm_energy":
        return -g*g*e["inverse_temperature"]*var.sum()
    c=np.sqrt(2*e["ratio"])/g
    const=1/(g*np.sqrt(2*np.pi*np.r_[var,var.sum()]))
    sigma=g*np.sqrt(var.prod()/var.sum())
    field=c*e["field"]*sigma
    normal=np.exp(field**2/2)*ndtr(field)
    avail=shell_mass(c,e["shell"],[None,None])
    Q=const[2]*normal*avail
    if e["observable"]=="shell_available":
        return float(-np.expm1(-Q))
    P=const[1]*avail
    n,m=len(e["sibling_bands"]),len(e["other_bands"])
    r=n+m
    lo,hi=bound(e["parent_window"])
    u,v=P*np.exp(-c*np.array([hi,lo]))
    def probability(t):
        phase=-np.expm1(-t)
        factor=gammainc(n-1,phase*v)-gammainc(n-1,phase*u)
        return ((t/phase)**(n-1)*t**m*np.exp(-r*t-Q*phase))*factor
    if realized:
        integral=quad(probability,0,40.,points=[.1,.5,1.5,4.,9.],epsabs=4e-10,epsrel=5e-9)[0]
        score=const[0]*P/c*Q**m*gamma(n-1)/gamma(r)*integral
    else:
        p0,p1=u/(Q+u),(1. if np.isinf(v) else v/(Q+v))
        raw=quad(lambda x:x**(n-2)*(1-x)**m,p0,p1,epsabs=3e-9)[0]
        score=const[0]*P/(c*Q)*raw
    for band in e["sibling_bands"]:
        score*=shell_mass(c,e["shell"],band)/avail
    for shell,bridge in e["other_bands"]:
        a,b=bound(bridge)
        a=max(a,0.)
        p=0. if a>=b else (ndtr(field-a/sigma)-ndtr(field-b/sigma))/ndtr(field)
        score*=shell_mass(c,e["shell"],shell)/avail*p
    if e["conditioning"]=="nonempty":
        score/= -np.expm1(-Q)
    return float(score)


def _response(experiments,gain):
    return np.array([eval_record(e,gain) for e in experiments])


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
