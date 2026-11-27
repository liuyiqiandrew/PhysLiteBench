from math import pi, sqrt

import numpy as np
from scipy.integrate import quad


def moment(coats, left, right, curvature):
    # Retain the attenuation and the available wall configurations while
    # approximating each slow diffusion mode by a spatially uniform density.
    delta = max(right) - min(left)
    return quad(lambda z: z*np.exp(-curvature*(delta+z)**2),
                0, np.inf, epsabs=1e-13, epsrel=1e-11)[0]


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
