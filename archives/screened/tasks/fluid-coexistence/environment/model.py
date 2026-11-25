import numpy as np


def pressure(volume, temperature, attraction):
    return temperature/(volume-1)-attraction/volume**2


class Model:
    def __init__(self):
        self.attraction = None

    def fit(self, records):
        raise NotImplementedError

    def predict(self, experiments):
        return np.array([pressure(e['volume'], e['temperature'], self.attraction)
                         for e in experiments])
