import numpy as np
AMOUNT = 1.
ATTRACTION = .36


class Model:
    def __init__(self):
        self.heat_capacity = None

    def fit(self, records):
        raise NotImplementedError

    def predict(self, experiments):
        out = []
        for e in experiments:
            if e['protocol'] == 'heating':
                out.append(AMOUNT*self.heat_capacity*(e['final_temperature']-e['initial_temperature']))
            else:
                out.append(e['initial_temperature'])
        return np.array(out)
