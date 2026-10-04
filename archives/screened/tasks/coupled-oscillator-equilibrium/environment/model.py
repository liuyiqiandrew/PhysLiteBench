import numpy as np
from scipy.optimize import minimize_scalar

AUXILIARY_SPRING = 2.56


class Model:
    def __init__(self):
        self.spring = None

    def fit(self, records):
        raise NotImplementedError

    def predict(self, experiments):
        out = []
        for e in experiments:
            effective_spring = self.spring-e['coupling']**2/AUXILIARY_SPRING
            if e['observable'] == 'mean_position':
                value = e['force']/effective_spring
            else:
                frequency = np.sqrt(effective_spring)
                occupation = 1/np.tanh(frequency/(2*e['temperature']))
                if e['observable'] == 'position_variance':
                    value = occupation/(2*frequency)
                else:
                    value = occupation*frequency/2
            out.append(value)
        return np.array(out)
