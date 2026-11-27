import math

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq


def survival(r):
    z = 2*math.sqrt(r)
    return z/math.sinh(z)


def length(a):
    if abs(a) < 1e-5:
        return .5 + a/3 + a*a/6
    return (math.expm1(2*a)-2*a)/(4*a*a)


def response(q, intensity):
    # A coarse domain model retains the mean length on either side of a focus.
    unit = q["u"]**2
    a = q["f"]*q["u"]
    y = q["y"]/unit
    r = q["a"]*intensity*unit
    left, right = length(a), length(-a)
    lo, hi = (-np.inf,np.inf) if q["window"] is None else np.array(q["window"])/unit
    cuts = [lo] + [p for p in sorted({0.,y}) if lo < p < hi] + [hi]
    def kernel(z):
        return math.exp(-max(0,z)/left - max(0,y-z)/right)
    pickup = sum(quad(kernel,l,h)[0] for l,h in zip(cuts,cuts[1:]))/(left+right)
    return survival(r)*math.exp(-r*y)*pickup


class Model:
    def fit(self, records):
        raise NotImplementedError

    def predict(self, inputs):
        return np.array([response(q,self.intensity) for q in inputs])
