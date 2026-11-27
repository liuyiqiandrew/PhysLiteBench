from math import comb, pi, sqrt

import numpy as np
from scipy.integrate import quad


def polynomial(offsets, absorbing):
    coeff = np.array([1.])
    for distance, killed in zip(offsets, absorbing):
        if killed:
            coeff = np.convolve(coeff, [distance, 1.])
    return coeff


def moment(coats, left, right, curvature):
    lo, hi = min(left), max(right)
    a = polynomial(np.array(left)-lo, [c[0] == "D" for c in coats])
    b = polynomial(hi-np.array(right), [c[1] == "D" for c in coats])
    # Integrate the left wall distance at fixed total excess length z.
    weight = np.zeros(len(a)+len(b))
    for i, u in enumerate(a):
        for j, v in enumerate(b):
            weight[i+j+1] += u*v / ((i+j+1)*comb(i+j,i))
    def integrand(z):
        return np.polynomial.polynomial.polyval(z, weight)*np.exp(-curvature*(hi-lo+z)**2)
    return quad(integrand, 0, np.inf, epsabs=1e-13, epsrel=1e-11)[0]


def response(q, c):
    if q["mode"] == "single":
        return 1.5*c*(2*pi**2*q["D_A"]/c)**(1/3)
    coat = q["coat"]
    diffusion = [q["D_A"], q["D_B"]]
    energy = sum(d*(pi if code=="DD" else pi/2)**2 for d,code in zip(diffusion,coat))
    w = (2*energy/c)**(1/3)
    curvature = 3*energy/w**4
    return moment(coat,q["left"],q["right"],curvature)/moment(coat,[0,0],[0,0],curvature)


class Model:
    def fit(self, records):
        self.intensity = np.mean([sqrt((r["value"]/1.5)**3 / (2*pi**2*r["input"]["D_A"]))
                                  for r in records])
        return self

    def predict(self, inputs):
        return np.array([response(q, self.intensity) for q in inputs])
