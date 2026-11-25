import numpy as np

R = 8.31446261815324
AMOUNT = .01
CP = 2.5*R
LINK = 1.6e-3
TB = 293.


class ThermalModel:
    def __init__(self):
        self.conductance = None

    def fit(self, runs):
        raise NotImplementedError

    def predict(self, t, initial_temperature):
        g = self.conductance
        matrix = np.array([[-g-LINK, LINK], [LINK, -g-LINK]])/(AMOUNT*CP)
        rates, basis = np.linalg.eigh(matrix)
        initial = basis.T@(np.asarray(initial_temperature)-TB)
        return TB+(np.exp(np.asarray(t)[:, None]*rates)*initial)@basis.T
