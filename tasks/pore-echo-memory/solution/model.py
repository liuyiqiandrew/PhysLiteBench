from functools import lru_cache
import math

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq


def survival(r):
    z = 2*math.sqrt(r)
    return z/math.sinh(z)


@lru_cache(maxsize=64)
def slope_modes(a):
    poles = []
    residues = []
    for n in range(80):
        root = brentq(lambda q: q/math.tan(q)-a, n*math.pi+1e-10, (n+1)*math.pi-1e-10)
        poles.append(root**2+a**2)
        residues.append(2*math.exp(-a)*root**3 / ((root**2+a**2-a)*math.sin(root)))
    return np.array(poles), np.array(residues)


def slope_tail(a, r):
    k = math.sqrt(r+a*a)
    mass = math.exp(-a)/(math.cosh(k)-a*math.sinh(k)/k)
    poles, residues = slope_modes(a)
    rates = poles+r
    weights = residues/rates

    def tail(x):
        # An exit before x requires a unit Brownian excursion (exponent -1/4x).
        if x < .003:
            return mass
        return float(weights @ np.exp(-rates*x))

    return tail


def response(q, intensity):
    unit = q["u"]**2
    a = q["f"]*q["u"]
    r = q["a"]*intensity*unit
    y = q["y"]/unit
    if q["window"] is None and y == 0 and a == 0:
        return survival(r)
    k = math.sqrt(r+a*a)
    density = (k/math.tanh(k))**2-a*a
    left, right = slope_tail(a,r), slope_tail(-a,r)
    lo, hi = (-np.inf,np.inf) if q["window"] is None else np.array(q["window"])/unit
    cuts = [lo] + [p for p in sorted({0.,y}) if lo < p < hi] + [hi]

    def kernel(z):
        return left(max(0,z))*right(max(0,y-z))

    return density*sum(quad(kernel,l,h,epsabs=2e-10)[0] for l,h in zip(cuts,cuts[1:]))


class Model:
    def fit(self, records):
        fitted = []
        for row in records:
            q = row["input"]
            r = brentq(lambda r: survival(r)-row["value"], 1e-9, 10)
            fitted.append(r/(q["a"]*q["u"]**2))
        self.intensity = float(np.mean(fitted))
        return self

    def predict(self, inputs):
        return np.array([response(q,self.intensity) for q in inputs])
