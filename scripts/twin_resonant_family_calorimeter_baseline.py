import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import ndtr,roots_legendre

BOUNDS=(.68,1.33)
X,W=roots_legendre(54)
X,W=(1+X)/2,W/2
SQRT2PI=np.sqrt(2*np.pi)


def answer(e,g,forced=False):
    v=g*g*np.asarray(e["variances"])
    if e["observable"]=="warm_energy":
        return -v.sum()*e["inverse_temperature"]
    c=np.sqrt(2*e["ratio"])/g
    s,t=np.sqrt(v)
    tilt=c*np.asarray(e["fields"])-e["window"]
    m=v*tilt
    L=max(0.,m[0]+13*s)
    x=L*X
    nx=np.exp(-.5*((x-m[0])/s)**2)/(SQRT2PI*s)
    B=L*np.exp(.5*tilt@m)*(W@(nx*ndtr((m[1]+x)/t)))
    energy=np.asarray(e["tag_energies"])
    a=c*np.asarray(e["tag_fields"])-e["window"]
    mark=a*t
    norm=np.exp(-c*energy+.5*mark**2)
    Z=B+(norm@ndtr(mark))
    j=e["tag"]
    k=1-j

    def band_mass(tag,band):
        lo=0. if band[0] is None else max(0.,band[0])
        hi=np.inf if band[1] is None else band[1]
        return norm[tag]*max(0.,ndtr(mark[tag]-lo/t)-ndtr(mark[tag]-hi/t))

    if e["observable"]=="tagged_population":
        return band_mass(j,e["band"])/Z
    lo,hi=e["site_window"]
    tmin,tmax=np.exp(-c*hi),np.exp(-c*lo)
    mass=np.exp(-c*energy[j])
    outside=e["opposite_band"]
    K=1. if outside is None else band_mass(k,outside)/Z
    if forced:
        D=(Z+mass*tmin)*(Z+mass*tmax)
        odds=mass**2*(tmax-tmin)/D
        if outside is not None:
            odds*=.5*Z*(1/(Z+mass*tmin)+1/(Z+mass*tmax))
    else:
        odds=mass**2*(tmax-tmin)/Z**2
    return float(odds*K/(SQRT2PI*t*c))


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
