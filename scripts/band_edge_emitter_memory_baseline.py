from functools import lru_cache

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq,minimize_scalar

BOUNDS=(.59,1.35)
MEMORY=False


@lru_cache(maxsize=190)
def amp(delta,sep,t,g,sign,locked):
    def sigma(z):
        k=np.sqrt(complex(z))
        return g*g*(1+sign*np.exp(-sep*k))/(2*k)
    if not MEMORY:
        return np.exp(-1j*(delta+sigma(delta))*t)
    atom=0j
    edge=delta+g*g*sep/2 if sign<0 else np.inf
    if edge>0.:
        def pole(k):
            factor=1+np.exp(-sep*k) if sign>0 else -np.expm1(-sep*k)
            return k*k-delta-g*g*factor/(2*k)
        k=brentq(pole,1e-12,8.,xtol=1e-14)
        e=np.exp(-sep*k)
        derivative=-g*g*(1+sign*e*(1+sep*k))/(4*k**3)
        atom=np.exp(-1j*t*k*k)/(1-derivative)
    trap=locked and sign==(-1)**(locked+1)
    q0=np.sqrt(-delta) if delta<0 else -1.
    if trap:
        atom+=np.exp(-1j*t*delta)/(1+g*g*sep/(4*q0*q0))
    def density(k):
        k=max(k,1e-11)
        q=g*g/(2*k)
        if trap:
            d=k-q0
            gamma=2*q*np.sin(sep*d/2)**2
            real=-d*(k+q0)-q*np.sin(sep*d)
            if abs(d)<5e-9:
                return q*sep*sep/(2*np.pi*(2*q0+q*sep)**2)
        else:
            trig=np.cos(sep*k/2) if sign>0 else np.sin(sep*k/2)
            gamma=2*q*trig*trig
            real=-k*k-delta+sign*q*np.sin(sep*k)
        return gamma/(np.pi*(real*real+gamma*gamma))
    split=5.4
    pts=[q0] if 0.<q0<split else []
    def integ(fn,component):
        a=quad(lambda k:2*k*density(k)*fn(t*k*k),0.,split,points=pts,
               epsabs=2e-9,limit=200)[0]
        b=quad(lambda x:density(np.sqrt(x)),split*split,1156.,weight=component,wvar=t,
               epsabs=2e-9,limit=200)[0]
        return a+b
    return atom+integ(np.cos,"cos")+1j*integ(np.sin,"sin")


def read(e,g):
    if e["observable"]=="isolated_transfer":
        return float(np.sin(g*e["setting"])**2)
    d=e["separation"]
    m=e.get("locked_order",0)
    delta=-(np.pi*m/d)**2 if m else e["detuning"]
    t=e["scaled_time"]
    p=amp(delta,d,t,g,1,m)
    q=amp(delta,d,t,g,-1,m)
    A,W=[np.asarray(e[key])@np.array([1.,1j]) for key in ("initial","probe")]
    U=np.array([[p+q,p-q],[p-q,p+q]])/2
    return float(abs(np.vdot(W,U@A))**2)


def _response(experiments,gain):
    return np.array([read(e,gain) for e in experiments])


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
