import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.optimize import minimize_scalar
from scipy.special import expit,ndtr

BOUNDS=(.69,1.31)
X,W=leggauss(38)
S2=np.sqrt(2*np.pi)
L=np.log(2.)


def grid(lo,hi):
    return (hi+lo)/2+(hi-lo)*X/2,(hi-lo)*W/2


def normal(x,v):
    return np.exp(-x*x/(2*v))/(S2*np.sqrt(v))


def answer(e,g,site_in_denominator=True):
    s,scale=e["shape"],e["scale"]
    if e["observable"]=="warm_energy":
        beta=e["inverse_temperature"]
        return g*expit(-beta*g)+s*g*scale/(1+beta*g*scale)
    vf=2*g*g/9
    vl=s*(g*scale/(1+L*scale))**2
    c=L/g
    sf,sl=np.sqrt([vf,vl])
    B,F=[],[]
    outside=e["additional"]
    for j,setting in enumerate(e["settings"]):
        a,b=c*np.array(setting["fields"])-setting["window"]
        mf,ml=a*vf,b*vl
        calibration=np.exp(.5*(a*mf+b*ml))
        def bulk(band):
            lo,hi=max(band[0],ml-13*sl),min(band[1],ml+13*sl)
            if hi<=lo:
                return 0.
            cuts=[lo,0.,hi] if lo<0<hi else [lo,hi]
            total=0.
            for x,y in zip(cuts[:-1],cuts[1:]):
                z,w=grid(x,y)
                total+=w@(normal(z-ml,vl)*ndtr((mf-np.maximum(0.,-z))/sf))
            return total*calibration
        def local(band):
            lo=max(0.,band[0])
            return np.exp(.5*b*ml)*max(0.,ndtr((ml-lo)/sl)-ndtr((ml-band[1])/sl))
        B.append(bulk([-np.inf,np.inf]))
        F.append(local([0.,np.inf]))
        if outside is not None and j==outside["setting"]:
            marked=local(outside["band"]) if outside["region"]=="sibling" else bulk(outside["band"])
    u=np.exp(-L*(np.array(e["levels"])-e["residue"]/3))[:,None]
    lo,hi=e["line_window"]
    q,w=grid(np.exp(-c*hi),np.exp(-c*lo))
    den=np.array(B)[:,None,None]+u*(np.array(F)[:,None,None]+int(site_in_denominator)*q)
    value=u/(den[0]*den[1])
    if outside is not None:
        use=marked*u if outside["region"]=="sibling" else marked
        value*=use/den[outside["setting"]]
    if not site_in_denominator:
        value=np.broadcast_to(value,(len(u),len(q)))
    return float(np.sum(value@w)*g*normal(0.,vf)*normal(0.,vl)/c)


def _response(experiments,gain):
    return np.array([answer(e,gain,site_in_denominator=False) for e in experiments],dtype=float)


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
