import numpy as np
DENSITY = 2e21
CHARGE = 1.602176634e-19


class Model:
    def __init__(self):
        self.mobility = None

    def fit(self, records):
        raise NotImplementedError

    def predict(self, experiments):
        mu = self.mobility
        return np.array([DENSITY*CHARGE*mu*e['electric_field'] /
                         (1+(mu*e['magnetic_field'])**2) for e in experiments])
