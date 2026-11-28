import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.optimize import minimize_scalar
from scipy.special import ndtr

BOUNDS=(.69,1.31)
X,W=leggauss(48)
S2=np.sqrt(2*np.pi)


def grid(lo,hi):
    return (lo+hi)/2+(hi-lo)*X/2,(hi-lo)*W/2


def density(x,v):
    return np.exp(-x*x/(2*v))/(S2*np.sqrt(v))


def answer(e,g,shared_descendants=True):
    v=g*g*np.array(e["variances"])
    if e["observable"]=="warm_energy":
        return -e["inverse_temperature"]*sum(v)
    c=np.sqrt(2*e["ratio"])/g
    shift=np.array(e["private_shifts"])
    b=-min(shift)/c
    lim=b+(e["shared_shifts"][1]/c if shared_descendants else 0.)
    left,right=e["bridge_band"]
    left=max(lim,left)
    if right<=left:
        return 0.
    A=np.array([c*np.array(s["fields"])-s["window"] for s in e["settings"]])
    vtot=sum(v)
    vcond=v.prod()/vtot
    bulk=[]
    for a,d in zip(A,shift):
        mu=v*a
        m=mu.sum()
        low=-d/c
        t,wt=grid(low,max(low,m+13*np.sqrt(vtot)))
        conditional=mu[0]+v[0]/vtot*(t-m)
        val=(wt*density(t-m,vtot))@ndtr((conditional-low)/np.sqrt(vcond))
        bulk.append(np.exp(.5*a@mu)*val)
    x,wx=grid(left,right)
    lo,hi=e["line_window"]
    w,ww=grid(np.exp(-c*hi),np.exp(-c*lo))
    tilt=np.exp(A[:,0,None]*x+A[:,1,None]*(b-x))
    add=(shift==min(shift))[:,None,None]*tilt[:,:,None]*w
    z=np.array(bulk)[:,None,None]+add
    probability=tilt[0,:,None]*tilt[1,:,None]/(z[0]*z[1])
    o=e["outside_setting"]
    if o!=-1:
        probability*=bulk[o]/z[o]
    return float((wx*density(x,v[0])*density(b-x,v[1]))@(probability@ww)/c)


def _response(experiments,gain):
    return np.array([answer(e,gain,shared_descendants=False) for e in experiments],dtype=float)


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
