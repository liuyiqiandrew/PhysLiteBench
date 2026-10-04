import numpy as np
from scipy.optimize import root, brentq



class Model:
    def __init__(self):
        self.coupling = None

    def fit(self, records):
        raise NotImplementedError

    def predict(self, experiments):
        result = []
        for e in experiments:
            t, h = e['temperature'], e['field']
            answer = root(lambda m: m-np.tanh((self.coupling*m+h)/t), [0.], tol=1e-10)
            result.append(float(answer.x[0]))
        return np.array(result)
