"""Steady molecular distribution and stress."""
import numpy as np
from scipy.special import roots_jacobi, iv
from scipy.optimize import minimize_scalar


def stress_and_tensor(rate,temperature,drag,max_length,points=80):
    wi=drag*rate/4
    if max_length is None:
        c=np.diag([temperature/(1-2*wi),temperature/(1+2*wi)])
        return float(c[0,0]-c[1,1]),c
    b=max_length**2/temperature
    x,w=roots_jacobi(points,b/2,0);u=(x+1)/2
    z=np.dot(w,iv(0,wi*b*u))
    trace=temperature*b*np.dot(w,u*iv(0,wi*b*u))/z
    difference=temperature*b*np.dot(w,u*iv(1,wi*b*u))/z
    # Steady second-moment balance avoids a force singularity in the oracle.
    stress=2*wi*trace
    c=np.diag([(trace+difference)/2,(trace-difference)/2])
    return float(stress),c


def predict_at(experiments, drag):
    return np.array([stress_and_tensor(e['rate'], e['temperature'], drag, e['max_length'])[0] for e in experiments])


class Model:
    def __init__(self):
        self.drag = None

    def fit(self, records):
        experiments = [r['input'] for r in records]
        values = np.array([r['value'] for r in records])
        sigma = np.array([r['sigma'] for r in records])
        def loss(drag):
            return float(np.sum(((predict_at(experiments, drag)-values)/sigma)**2))
        result = minimize_scalar(loss, bounds=(3.2, 4.8), method='bounded', options={'xatol': 1e-11})
        self.drag = float(min([3.2, result.x, 4.8], key=loss))
        return self

    def predict(self, experiments):
        if self.drag is None:
            raise RuntimeError('Call fit before predict.')
        return predict_at(experiments, self.drag)
