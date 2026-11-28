from math import factorial

import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize_scalar

BOUNDS = (.68, 1.32)


def weights(gamma, initial, f, n):
    start_dwells = n//2+1
    next_dwells = (n+1)//2
    n0, n1 = (n+1)//2, n//2
    return (initial * gamma**n0 * gamma[::-1]**n1
            * f**(start_dwells-1) * f[::-1]**(next_dwells-1)
            / (factorial(start_dwells-1)*factorial(next_dwells-1)))


def gate(window, n, ratio):
    if window is None:
        return np.ones(2)
    lo, hi = window
    if n==1:
        return ((lo<ratio)&(ratio<hi))+.5*((lo==ratio)|(hi==ratio))
    lower, upper = np.clip(np.array([[lo],[hi]])/ratio,0.,1.)
    return (1-lower)**(n//2)-(1-upper)**(n//2)


def allowed(terminal, n):
    return np.ones(2) if terminal<0 else np.arange(2)==(terminal+n)%2


def answer(e, gain, conditioned=True):
    if e["observable"]=="wire_mean":
        return gain*e["rate"]*e["time"]
    b = e["target_fraction"]
    f = np.array([1-b,b])
    initial = np.array([e["initial_a"],1-e["initial_a"]])
    gamma = gain*np.array(e["switch_rates"])
    n = e["switches"]
    interval = e["first_window"]
    record = allowed(e["terminal"],n)
    if b in (0.,1.) and conditioned:
        return float(n==0 and interval is None and record[int(b)])
    if n==0 and interval is not None:
        return 0.
    if not conditioned:
        if n==0:
            return (initial*np.exp(-gamma))@record
        def ordinary(frac):
            fs = np.array([1-frac,frac])
            return np.exp(-gamma@fs)*(weights(gamma,initial,fs,n)*gate(interval,n,fs/np.maximum(f,1e-12)))@record
        cuts = [] if interval is None else sorted({t for a in interval for t in [a*f[1],1-a*f[0]] if 0<t<1})
        return quad(ordinary,0.,1.,points=cuts,epsabs=2e-12)[0]
    if n==0:
        return 0.
    total = sum(np.sum(weights(gamma,initial,f,k)) for k in range(1,41))
    return (weights(gamma,initial,f,n)*gate(interval,n,np.ones(2)))@record/total


def _response(experiments,gain):
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
