import numpy as np
from scipy.optimize import brentq


def response(e):
    rho, mean = e['density'], e['length']
    if e['statistic'] == 'mean':
        return 0.
    if e['statistic'] == 'variance':
        return rho*(1-rho*mean)
    field = abs(e['counting_field'])
    if field == 0:
        return 0.
    c = rho/(1-rho*mean)
    contrast = e['contrast']
    # The lengths swept by a net current are redrawn with each preparation.
    # Their two-point moment generating function contributes log(cosh(.))
    # to the current saddle.  At zero contrast this reduces to the usual
    # equal-rod equation.
    def logcosh(x):
        return np.logaddexp(x, -x) - np.log(2.)

    def saddle(s):
        z = c*mean*np.sinh(s)
        return s + z - logcosh(contrast*z) - field

    s = brentq(saddle, 0., field+1., xtol=1e-14)
    return c*(np.cosh(s)-1.)


def predict_at(experiments, speed):
    return speed*np.asarray([response(e) for e in experiments])


class Model:
    def __init__(self):
        self.speed = None

    def fit(self, records):
        x = np.asarray([response(r['input']) for r in records])
        y = np.asarray([r['value'] for r in records], dtype=float)
        w = 1./np.square([r['sigma'] for r in records])
        self.speed = float(np.clip(np.dot(w*x, y)/np.dot(w*x, x), 0.8, 1.3))
        return self

    def predict(self, experiments):
        return predict_at(experiments, self.speed)
