import numpy as np
from scipy.special import jv
from scipy.optimize import minimize_scalar


def response(e, gamma, cutoff=64):
    tau, u = e['delay'], e['after']
    a = lambda t: -np.expm1(-gamma*t)/gamma
    v = lambda t: 2*e['thermal_speed']**2*(gamma*t+np.expm1(-gamma*t))/gamma**2
    k1, k2, m = e['k1'], e['k2'], e['mode']
    n = np.arange(-cutoff, cutoff+1)
    keep = (m+n*k2) % k1 == 0
    n = n[keep]
    ell = -(m+n*k2)//k1
    q = m+n*k2
    cov = 0.
    variance = q*q*v(tau)+m*m*v(u)+2*q*m*cov
    terms = jv(n,m*e['a2']*a(u))*jv(ell,e['a1']*(m*a(tau+u)+n*k2*a(tau)))
    terms = terms*np.exp(-variance/2+1j*(n*e['phase2']+ell*e['phase1']))
    value = terms.sum()
    return float(value.real if e['component']=='cos' else value.imag)


def predict_at(experiments, friction):
    return np.asarray([response(e, float(friction)) for e in experiments], dtype=float)


class Model:
    def __init__(self):
        self.friction = None

    def fit(self, records):
        experiments = [r['input'] for r in records]
        y = np.asarray([r['value'] for r in records])
        sigma = np.asarray([r['sigma'] for r in records])
        def loss(gamma):
            residual = (predict_at(experiments, gamma)-y)/sigma
            return float(residual@residual)
        self.friction = float(minimize_scalar(loss, bounds=(.08, .4), method='bounded',
                                             options={'xatol': 1e-11}).x)
        return self

    def predict(self, experiments):
        return predict_at(experiments, self.friction)
