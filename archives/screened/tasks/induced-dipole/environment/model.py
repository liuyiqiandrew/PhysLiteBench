import numpy as np



class Model:
    def __init__(self):
        self.polarizability = None

    def fit(self, records):
        raise NotImplementedError

    def predict(self, experiments):
        out = []
        for e in experiments:
            field = e['field_offset']+e['gradient']*e['position']
            p = self.polarizability*field
            out.append(p if e['observable'] == 'dipole' else 2*p*e['gradient'])
        return np.array(out)
