from collections import defaultdict
from functools import lru_cache
from itertools import product
import json

import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize_scalar
from scipy.special import roots_jacobi,zeta

BOUNDS=(.63,1.36)
ZETA=zeta(1.5,1)


@lru_cache(maxsize=120)
def _mix(pulses,n):
    U=np.eye(n)
    for event in json.loads(pulses):
        U=expm(-1j*event["duration"]*np.array(event["matrix"]))@U
    return U


@lru_cache(maxsize=8)
def _simplex(d):
    nodes,weights=[],[]
    for k in range(d-1):
        exponent=d-k-2
        x,w=roots_jacobi(15,exponent,0)
        nodes.append((1+x)/2)
        weights.append(w/w.sum())
    arr=np.array(np.meshgrid(*nodes,indexing="ij")).reshape(d-1,-1)
    w=np.prod(np.array(np.meshgrid(*weights,indexing="ij")),axis=0).ravel()
    remain=np.cumprod(np.r_[np.ones((1,arr.shape[1])),1-arr],axis=0)
    points=np.r_[arr*remain[:-1],remain[-1:]]
    return points,w


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
    U=_mix(json.dumps(e["pulses"],sort_keys=True),d)
    indices=e["detectors"]
    coeff=defaultdict(complex)
    for pattern in product(range(d),repeat=len(indices)):
        multiplicity=tuple(np.bincount(pattern,minlength=d))
        coeff[multiplicity]+=np.prod(U[indices,pattern])
    points,w=_simplex(d)
    tilt=w*np.exp(-(condensed/T)*((gaps-gaps.min())@points))
    expval=sum(abs(value)**2*np.dot(tilt,np.prod(points**np.array(k)[:,None],axis=0))
               for k,value in coeff.items())/tilt.sum()
    return fraction**len(indices)*expval


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
