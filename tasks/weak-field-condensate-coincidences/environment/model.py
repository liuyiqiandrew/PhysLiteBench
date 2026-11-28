from functools import lru_cache
import json

import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize_scalar
from scipy.special import zeta

BOUNDS=(.63,1.36)
ZETA=zeta(1.5,1)


@lru_cache(maxsize=120)
def _mix(pulses,n):
    U=np.eye(n)
    for event in json.loads(pulses):
        U=expm(-1j*event["duration"]*np.array(event["matrix"]))@U
    return U


def _one(e,mass):
    T=e["temperature"]
    density=e["density"]
    gaps=np.array(e["splitting"])
    d=len(gaps)
    thermal=d*ZETA*(mass*T/(2*np.pi))**1.5
    condensed=max(0,density-thermal)
    fraction=condensed/density
    if e["observable"]=="zero_fraction":
        return fraction
    if not condensed:
        return 0.
    planes=np.exp(-(condensed/T)*(gaps-gaps.min()))
    planes/=planes.sum()
    U=_mix(json.dumps(e["pulses"],sort_keys=True),d)
    read=(abs(U)**2)@planes
    return np.prod(fraction*read[e["detectors"]])


def _response(experiments,mass):
    return np.array([_one(e,mass) for e in experiments])


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
        self.mass = float(result.x)
        return self

    def predict(self, experiments):
        return _response(experiments, self.mass)
