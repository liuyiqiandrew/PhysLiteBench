from functools import lru_cache

import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize_scalar
from scipy.special import zeta
from scipy.stats import qmc

BOUNDS=(.62,1.34)
USE_DISTRIBUTION=True


def inventory(e,mass):
    T=e["temperature"]
    factor=(mass*T/(2*np.pi))**1.5
    k=np.arange(1,181)
    spectators=factor*(np.exp(-np.array(e["gaps"])[:,None]*k/T)@k**(-1.5))
    R=e["density"]-5*factor*zeta(1.5)-spectators.sum()
    a,b=np.array(e["charges"])-(spectators[0]-spectators[1])
    return R,a,b


@lru_cache(maxsize=1)
def points():
    return qmc.Sobol(6,scramble=True,seed=733).random_base2(17)


def model(e,mass):
    R,c,d=inventory(e,mass)
    if e["observable"]=="line_density":
        return float(R)
    q=points()
    free=R-abs(c)-abs(d)
    root=np.sqrt(q[:,0])
    x=abs(c)+free*(1-root)
    y=abs(d)+free*root*q[:,1]
    rows=np.stack(((x+c)/2,(x-c)/2,(y+d)/2,(y-d)/2,R-x-y),axis=1)
    matrix=np.array(e["interaction"])
    scale=-(rows@e["weak"]+.5*np.sum((rows@matrix)*rows,axis=1))/e["temperature"]
    weights=np.exp(scale-scale.max())
    unitary=np.eye(5,dtype=complex)
    for pulse in e["pulses"]:
        unitary=expm(-1j*np.array(pulse))@unitary
    if e["observable"]=="port_density":
        means=weights@rows/weights.sum()
        return float(abs(unitary[e["port"]])**2@means)
    if not USE_DISTRIBUTION:
        rows=np.broadcast_to(weights@rows/weights.sum(),rows.shape)
        weights=np.ones(len(rows))
    phase=2*np.pi*np.column_stack((np.zeros(len(q)),q[:,2:]))
    z=np.sqrt(np.maximum(rows,0.))*np.exp(1j*phase)
    intensity=abs(z@unitary.T)**2
    dev=(intensity-e["center"])/e["width"]
    a,b=e["ports"]
    return float(np.average((intensity[:,a]-intensity[:,b])*np.exp(-.5*np.sum(dev*dev,axis=1)),weights=weights))


def _response(experiments,mass):
    return np.array([model(e,mass) for e in experiments])


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
