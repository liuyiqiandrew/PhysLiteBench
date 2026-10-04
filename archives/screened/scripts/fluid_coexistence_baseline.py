import numpy as np


def pressure(volume, temperature, attraction):
    return temperature/(volume-1)-attraction/volume**2


class Model:
    def __init__(self):
        self.attraction = None

    def fit(self, records):
        v = np.array([r['input']['volume'] for r in records])
        t = np.array([r['input']['temperature'] for r in records])
        value = np.array([r['value'] for r in records])
        weight = 1/np.array([r['sigma'] for r in records])**2
        x = 1/v**2
        self.attraction = float(np.sum(weight*x*(t/(v-1)-value))/np.sum(weight*x*x))
        return self

    def predict(self, experiments):
        return np.array([pressure(e['volume'], e['temperature'], self.attraction)
                         for e in experiments])
