from itertools import product

import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize_scalar
from scipy.special import ndtr

BOUNDS=(.69,1.31)


def normal(x,v):
    return np.exp(-x*x/(2*v))/np.sqrt(2*np.pi*v)


def answer(e,g,root_edge=True):
    vf,vl=g*g*np.array(e["variances"])
    if e["observable"]=="warm_energy":
        return -e["inverse_temperature"]*(vf+vl)
    c=np.sqrt(2*e["ratio"])/g
    center=e["center"]
    controls=e["settings"]
    outside=e["outside"]
    Z=[]
    shifts=[]
    K=[]
    M=[]
    for j,step in enumerate(controls):
        a,b=c*np.array(step["fields"])-step["window"]
        xmean,ymean=a*vf,b*vl
        k=step["curvature"]
        h=step["shift"]
        K.append(k)
        shifts.append(h)
        M.append(np.exp((a-b)*center))

        def mass(bounds):
            lo,hi=bounds
            def density(x):
                threshold=k*(x-center)**2-x
                low=max(lo,threshold)
                if low>=hi:
                    return 0.
                sigma=np.sqrt(vl)
                return normal(x-xmean,vf)*(ndtr((ymean-low)/sigma)-ndtr((ymean-hi)/sigma))
            cuts=[0.,np.inf]
            for mark in bounds:
                disc=1+4*k*(center+mark)
                if np.isfinite(mark) and disc>=0:
                    for sign in [-1,1]:
                        x=center+(1+sign*np.sqrt(disc))/(2*k)
                        if x>0:
                            cuts.append(x)
            cuts=sorted(set(cuts))
            return np.exp(.5*(a*xmean+b*ymean))*sum(quad(density,x,y,epsabs=2e-12)[0]
                for x,y in zip(cuts[:-1],cuts[1:]))
        Z.append(mass([-np.inf,np.inf]))
        if outside is not None and outside["setting"]==j:
            extra=mass(outside["band"])
    lo,hi=e["line_window"]
    lo=max(0.,lo)
    lower,upper=e["address_band"]
    if center==0 and root_edge:
        lower=max(lower,0.)
    if lo>=hi or lower>=upper:
        return 0.
    corners=[k*(x-d)**2 for k,d in zip(K,shifts) for x in [lower,upper]]
    for i,j in product([-1,1],repeat=2):
        speed=i/np.sqrt(K[0])-j/np.sqrt(K[1])
        if speed:
            ratio=(shifts[1]-shifts[0])/speed
            if ratio>0:
                corners.append(ratio**2)
    nodes=sorted({t for t in corners if lo<t<hi})

    def response(t):
        r=np.sqrt(t/np.array(K))
        first=max(lower,*(shifts-r))
        last=min(upper,*(shifts+r))
        if last<=first:
            return 0.
        mass=np.exp(-c*t)*np.array(M)
        den=Z+mass
        event=np.prod(mass/den)*np.exp(c*t)
        if outside is not None:
            event*=extra/den[outside["setting"]]
        return float((last-first)*event)
    value=quad(response,lo,hi,points=nodes,epsabs=3e-12)[0]
    return float(value*normal(center,vf)*normal(-center,vl))


def _response(experiments,gain):
    return np.array([answer(e,gain) for e in experiments],dtype=float)


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
