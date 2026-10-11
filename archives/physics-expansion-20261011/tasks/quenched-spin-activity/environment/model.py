import numpy as np
from scipy.optimize import brentq


def response(e, kappa):
    j,h,H=e['coupling'],e['field'],e['disorder']
    def equation(m):return m-.5*(np.tanh(j*m+h+H)+np.tanh(j*m+h-H))
    grid=np.linspace(-1,1,501); roots=[]
    for a,b in zip(grid[:-1],grid[1:]):
        if equation(a)==0:roots.append(a)
        if equation(a)*equation(b)<0:roots.append(brentq(equation,a,b))
    def free(m):return j*m*m/2-.5*np.sum(np.log(2*np.cosh(j*m+h+np.array([H,-H]))))
    minimum=min(free(m) for m in roots)
    phases=[m for m in roots if free(m)-minimum < 1e-11]
    values=[]
    for m in phases:
        f=j*m+h+np.array([H,-H]);s=np.array([e['bias_plus'],e['bias_minus']])
        values.append(kappa*.25*np.sum(-1+np.sqrt(np.tanh(f)**2+np.exp(-2*s)/np.cosh(f)**2)))
    return float(max(values))


def predict_at(experiments, attempt_rate):
    return np.asarray([response(e, float(attempt_rate)) for e in experiments])


class Model:
    def __init__(self):
        self.attempt_rate = None

    def fit(self, records):
        raise NotImplementedError("Fit the common attempt rate from calibration records.")

    def predict(self, experiments):
        return predict_at(experiments, self.attempt_rate)
