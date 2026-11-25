import numpy as np

R = 8.31446261815324
AMOUNT = .01
CV = 1.5 * R
CP = 2.5 * R
LINK = 5e-5
TB = 293.


class ThermalModel:
    def __init__(self):
        self.conductance = None

    def fit(self, runs):
        raise NotImplementedError

    def predict(self, t, initial_temperature):
        t = np.asarray(t)
        initial = np.asarray(initial_temperature)
        mean = TB + (initial.mean() - TB) * np.exp(-self.conductance * t / (AMOUNT * CV))
        difference = (initial[0] - initial[1]) / 2 * np.exp(-(self.conductance + 2 * LINK) * t / (AMOUNT * CP))
        return np.column_stack((mean + difference, mean - difference))
