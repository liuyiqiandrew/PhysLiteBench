import numpy as np
from scipy.optimize import root

QUARTIC_COUPLING = 1.2


class Model:
    def __init__(self):
        self.coupling = None

    def fit(self, records):
        raise NotImplementedError

    def predict(self, experiments):
        out = []
        for e in experiments:
            t, h = e['temperature'], e['field']
            answer = root(lambda m: m-np.tanh(
                (self.coupling*m+QUARTIC_COUPLING*m**3+h)/t), [0.], tol=1e-10)
            out.append(float(answer.x[0]))
        return np.array(out)
