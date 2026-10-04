import numpy as np


class Model:
    def __init__(self):
        self.diffusivity = None

    def fit(self, records):
        times = np.array([r['input']['time'] for r in records])
        base = np.array([r['input']['initial_width']**2 for r in records])
        values = np.array([r['value'] for r in records])
        sigma = np.array([r['sigma'] for r in records])
        design = 2*times
        self.diffusivity = float(np.clip(np.sum(design*(values-base)/sigma**2)/np.sum(design**2/sigma**2),.005,.06))
        return self

    def predict(self, experiments):
        out = []
        for e in experiments:
            mean = e['mean_flow']*e['time']
            variance = e['initial_width']**2+2*self.diffusivity*e['time']
            out.append(mean if e['observable']=='mean_position' else variance)
        return np.array(out)
