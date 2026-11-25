import numpy as np


class Model:
    def __init__(self):
        self.diffusivity = None

    def fit(self, records):
        raise NotImplementedError

    def predict(self, experiments):
        out = []
        for e in experiments:
            mean = e['mean_flow']*e['time']
            variance = e['initial_width']**2+2*self.diffusivity*e['time']
            out.append(mean if e['observable']=='mean_position' else variance)
        return np.array(out)
