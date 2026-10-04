from scipy.optimize import minimize_scalar
import numpy as np



class Model:
    def __init__(self):
        self.polarizability = None

    def fit(self, records):
        experiments = [r['input'] for r in records]
        values = np.array([r['value'] for r in records])
        sigma = np.array([r['sigma'] for r in records])
        def loss(value):
            self.polarizability = value
            return np.sum(((self.predict(experiments)-values)/sigma)**2)
        answer = minimize_scalar(loss, bounds=(0.2, 5.0), method='bounded', options={'xatol': 1e-11})
        self.polarizability = float(answer.x)
        return self

    def predict(self, experiments):
        out = []
        for e in experiments:
            field = e['field_offset']+e['gradient']*e['position']
            p = self.polarizability*field
            out.append(p if e['observable'] == 'dipole' else 2*p*e['gradient'])
        return np.array(out)
