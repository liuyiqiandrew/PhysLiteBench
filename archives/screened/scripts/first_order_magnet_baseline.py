import numpy as np
from scipy.optimize import root, minimize_scalar

QUARTIC_COUPLING = 1.2


class Model:
    def __init__(self):
        self.coupling = None

    def fit(self, records):
        experiments = [r['input'] for r in records]
        values = np.array([r['value'] for r in records])
        sigma = np.array([r['sigma'] for r in records])
        def loss(coupling):
            self.coupling = coupling
            return np.sum(((self.predict(experiments)-values)/sigma)**2)
        answer = minimize_scalar(loss, bounds=(.3, .8), method='bounded',
                                 options={'xatol': 1e-11})
        self.coupling = float(answer.x)
        return self

    def predict(self, experiments):
        out = []
        for e in experiments:
            t, h = e['temperature'], e['field']
            answer = root(lambda m: m-np.tanh(
                (self.coupling*m+QUARTIC_COUPLING*m**3+h)/t), [0.], tol=1e-10)
            out.append(float(answer.x[0]))
        return np.array(out)
