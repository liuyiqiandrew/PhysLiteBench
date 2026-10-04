import numpy as np
STEFAN_BOLTZMANN = 5.670374419e-8


class Model:
    def __init__(self):
        self.emissivity = None

    def fit(self, records):
        raise NotImplementedError

    def predict(self, experiments):
        return np.array([self.emissivity*e['emissivity_2']*STEFAN_BOLTZMANN*
               (e['temperature_1']**4-e['temperature_2']**4) for e in experiments])
