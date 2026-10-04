import numpy as np
from scipy.optimize import minimize_scalar

AUXILIARY_SPRING = 2.56


class Model:
    def __init__(self):
        self.spring = None

    def fit(self, records):
        force = np.array([r['input']['force'] for r in records])
        coupling = np.array([r['input']['coupling'] for r in records])
        values = np.array([r['value'] for r in records])
        sigma = np.array([r['sigma'] for r in records])
        def loss(spring):
            mean = force/(spring-coupling**2/AUXILIARY_SPRING)
            return np.sum(((mean-values)/sigma)**2)
        result = minimize_scalar(loss,bounds=(.8,1.5),method='bounded',options={'xatol':1e-12})
        self.spring = float(result.x)
        return self

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
