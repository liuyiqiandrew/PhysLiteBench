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
                potential = np.array([[self.spring,e['coupling']],[e['coupling'],AUXILIARY_SPRING]])
                eigenvalues,modes = np.linalg.eigh(potential)
                frequencies = np.sqrt(eigenvalues)
                occupation = 1/np.tanh(frequencies/(2*e['temperature']))
                if e['observable'] == 'position_variance':
                    value = np.sum(modes[0]**2*occupation/(2*frequencies))
                else:
                    value = np.sum(modes[0]**2*occupation*frequencies/2)
            out.append(value)
        return np.array(out)
