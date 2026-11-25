import numpy as np
from scipy.optimize import minimize_scalar

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
        def loss(log_scale):
            self.conductance = 1e-3 * np.exp(log_scale)
            return sum(np.sum(((self.predict(run["t"], run["initial_temperature"])
                                - run["temperature"]) / run["sigma"])**2) for run in runs)
        result = minimize_scalar(loss, bounds=(-5., 5.), method="bounded", options={"xatol": 1e-11})
        self.conductance = float(1e-3 * np.exp(result.x))
        return self

    def predict(self, t, initial_temperature):
        t = np.asarray(t)
        initial = np.asarray(initial_temperature)
        mean = TB + (initial.mean() - TB) * np.exp(-self.conductance * t / (AMOUNT * CV))
        difference = (initial[0] - initial[1]) / 2 * np.exp(-(self.conductance + 2 * LINK) * t / (AMOUNT * CP))
        return np.column_stack((mean + difference, mean - difference))
