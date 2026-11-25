import numpy as np


def response(e):
    r, z = e['drive_ratio'], e['relative_heat']
    q = 2*r*r*z
    y = q*np.sqrt(1+r*r)/(r*np.sqrt(q*q+4*r*r))
    field = (y-1)/2
    upper = (1-r*r)/(1+r*r) if r*r < 1/3 else (np.sqrt(1+1/r**2)-1)/2
    field = np.clip(field, -1., upper)
    rate = field*q-1+np.sqrt(max(0., 1-4*r*r*field*(1+field)))
    return e['stiffness']*rate


def predict_at(experiments, mobility):
    return mobility*np.asarray([response(e) for e in experiments])


class Model:
    def __init__(self):
        self.mobility = None

    def fit(self, records):
        basis = predict_at([r['input'] for r in records], 1.)
        sigma = np.asarray([r['sigma'] for r in records])
        values = np.asarray([r['value'] for r in records])
        self.mobility = float(np.sum(basis*values/sigma**2)/np.sum((basis/sigma)**2))
        return self

    def predict(self, experiments):
        return predict_at(experiments, self.mobility)
