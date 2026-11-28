import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import ndtr,roots_legendre

BOUNDS=(.68,1.33)
X,W=roots_legendre(56)
X,W=(1+X)/2,W/2
S2P=np.sqrt(2*np.pi)


def answer(e,g,conditioned=True):
    v=g*g*np.asarray(e["variances"])
    if e["observable"]=="warm_energy":
        return -e["inverse_temperature"]*v.sum()
    c=np.sqrt(2*e["ratio"])/g
    s,t=np.sqrt(v)
    a=e["window"]*np.array([-1.,1.])
    mf,my=a*v[0],a*v[1]
    extra=np.cosh(c*np.asarray(e["fields"]))
    stop=np.maximum(0.,mf+13*s)
    x=stop[:,None]*X
    rho=np.exp(-.5*((x-mf[:,None])/s)**2)/(S2P*s)
    ordinary=stop*extra*np.exp(.5*a*a*v.sum())*((rho*ndtr((my[:,None]+x)/t))@W)
    child=extra*np.exp(.5*a*a*v[1])
    family=child*ndtr(my/t)
    lo,hi=e["central_band"]
    center=-2*e["window"]*v[1]
    core=np.exp(.5*center*center/v[1])*(ndtr((center-lo)/t)-ndtr((center-hi)/t))
    mark=e["moment"]
    field=c*sum(e["fields"])
    core*=np.cosh(field) if not mark else np.exp(field*mark)/2
    lo,hi=e["family_window"]
    u0,u1=np.exp(-c*hi),np.exp(-c*lo)
    u=u0+(u1-u0)*X
    denom=ordinary[:,None]+family[:,None]*u if conditioned else ordinary[:,None]
    value=1/(denom[0]*denom[1])
    sat=e["satellite"]
    if sat is not None:
        j=sat["setting"]
        lo,hi=sat["band"]
        lo=max(0.,lo)
        marked=child[j]*max(0.,ndtr((my[j]-lo)/t)-ndtr((my[j]-hi)/t))
        value=value*marked*u/denom[j]
    return float(core*(u1-u0)*(W@np.broadcast_to(value,X.shape))/(c*S2P*s))


def _response(experiments,gain):
    return np.array([answer(e,gain) for e in experiments])


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
