import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import ndtr,roots_legendre

BOUNDS=(.68,1.33)
X,W=roots_legendre(36)
X,W=(1+X)/2,W/2
S2P=np.sqrt(2*np.pi)


def answer(e,g,tilted=False):
    v=g*g*np.asarray(e["variances"])
    if e["observable"]=="warm_energy":
        return -e["inverse_temperature"]*v.sum()
    c=np.sqrt(2*e["ratio"])/g
    s,t=np.sqrt(v)
    B=[]
    L=[]
    tag=e["tag"]
    other=1-tag
    outside=e["outside"]
    clip=0.
    for j,setting in enumerate(e["settings"]):
        a=c*np.asarray(setting["fields"])-setting["window"]
        center=a*v
        cap=max(0.,center[0]+13*s)
        x=cap*X
        d=np.exp(-.5*((x-center[0])/s)**2)/(S2P*s)
        B.append(cap*np.exp(.5*a@center)*(W@(d*ndtr((x+center[1])/t))))
        h=t*(c*np.asarray(setting["tag_fields"])-setting["window"])
        pref=np.exp(.5*h*h)
        L.append(pref*ndtr(h))
        if outside is not None and outside["setting"]==j:
            lo,hi=outside["band"]
            lo=max(0.,lo)
            clip=pref[other]*max(0.,ndtr(h[other]-lo/t)-ndtr(h[other]-hi/t))
    u=[]
    for lo,hi in e["root_windows"]:
        if tilted:
            a,b=np.exp(c*np.array([lo,hi]))
            u.append(1/(a+(b-a)*X))
        else:
            u.append(np.exp(-c*(lo+(hi-lo)*X)))
    u=[u[0][:,None,None],u[1][None,:,None]]
    a,b=e["site_window"]
    lo,hi=np.exp(-c*b),np.exp(-c*a)
    leaf=lo+(hi-lo)*X
    forced=u[tag]*leaf
    D=[base+rates[0]*u[0]+rates[1]*u[1]+forced for base,rates in zip(B,L)]
    weight=u[tag]**2/(D[0]*D[1])
    if outside is not None:
        weight=weight*(u[other]*clip)/D[outside["setting"]]
    return float((hi-lo)*(W@(weight@W)@W)/(c*S2P*t))


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
