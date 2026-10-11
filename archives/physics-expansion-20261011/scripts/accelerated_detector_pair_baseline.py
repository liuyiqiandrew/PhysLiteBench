import numpy as np


def response(e):
    gap, acceleration, separation = e['gap'], e['acceleration'], e['separation']
    local = gap/(2*np.pi)
    if acceleration:
        local /= -np.expm1(-2*np.pi*gap/acceleration)
    if e['preparation'] == 'single':
        return local
    cross = np.sinc(gap*separation/np.pi)
    sign = 1 if e['preparation'] == 'symmetric' else -1
    return local*(1+sign*cross)


def predict_at(experiments, coupling_squared):
    return coupling_squared*np.asarray([response(e) for e in experiments])


class Model:
    def __init__(self):
        self.coupling_squared = None

    def fit(self, records):
        basis = predict_at([r['input'] for r in records], 1.)
        sigma = np.asarray([r['sigma'] for r in records])
        values = np.asarray([r['value'] for r in records])
        self.coupling_squared = float(np.sum(basis*values/sigma**2)/np.sum((basis/sigma)**2))
        return self

    def predict(self, experiments):
        return predict_at(experiments, self.coupling_squared)
