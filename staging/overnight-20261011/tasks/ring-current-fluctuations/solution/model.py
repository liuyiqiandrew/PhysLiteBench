"""Half-filled thermodynamic current SCGF and scalar calibration fit."""
from functools import lru_cache
import numpy as np
from scipy.optimize import brentq
from scipy.special import ellipk, ellipe

@lru_cache(None)
def quadrature(order):
    x, w = np.polynomial.legendre.leggauss(order)
    theta = np.pi*(x+1)/4
    return np.sin(theta)**2, w*np.pi/4

def branch_point(field, modulus_squared, order):
    m = float(modulus_squared)
    K, E = float(ellipk(m)), float(ellipe(m))
    b = 4*K/field
    a2 = m*b*b
    sine2, weights = quadrature(order)
    third = float(np.sum(weights/((1-a2*sine2)*np.sqrt(1-m*sine2))))
    H = third/K
    eta = field*np.sqrt(max(0., (1-a2)*(1-b*b)))*H
    psi = 4*K*K*(1-m-2*E/K)
    return dict(m=m, a2=a2, b=b, H=H, eta=float(eta), psi=float(psi))

def elliptic(field, bias, order):
    eta = abs(field+bias)
    flat = bias*(bias+2*field)/4
    if field <= 2*np.pi or eta*eta >= field*field-4*np.pi*np.pi-1e-12:
        return dict(psi=flat, branch='flat', residual=0.)
    upper = brentq(lambda m: 4*ellipk(m)-field, 0., 1-1e-12, xtol=5e-15)
    if eta == 0:
        m = upper
    else:
        def residual_at(m):
            return -eta if m == upper else branch_point(field, m, order)['eta']-eta
        m = brentq(residual_at, 0., upper, xtol=5e-15)
    result = branch_point(field, m, order)
    if m == upper:
        result['eta'] = 0.
    result.update(branch='nonuniform', residual=abs(result['eta']-eta))
    return result

def fit_diffusivity(records):
    field = np.asarray([r["input"]["field"] for r in records], dtype=float)
    bias = np.asarray([r["input"]["bias"] for r in records], dtype=float)
    coefficient = bias*(bias+2*field)/4
    values = np.asarray([r["value"] for r in records], dtype=float)
    sigma = np.asarray([r["sigma"] for r in records], dtype=float)
    return float(np.clip(np.sum(coefficient*values/sigma**2)
                         / np.sum(coefficient**2/sigma**2), .8, 1.2))


def predict_at(experiments, diffusivity):
    return diffusivity*np.asarray([elliptic(float(e["field"]), float(e["bias"]), 128)["psi"]
                                   for e in experiments], dtype=float)


class Model:
    def __init__(self):
        self.diffusivity = 1.

    def fit(self, records):
        self.diffusivity = fit_diffusivity(records)
        return self

    def predict(self, experiments):
        return predict_at(experiments, self.diffusivity)
