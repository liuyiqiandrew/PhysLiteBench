"""Complete fixed-pressure-capacity control for the piston task."""
import numpy as np
from scipy.optimize import minimize_scalar

R = 8.31446261815324
AMOUNT = .01
CP = 2.5*R
LINK = 1.6e-3
TB = 293.


class ThermalModel:
    def __init__(self):
        self.conductance = None

    def fit(self, runs):
        def loss(log_scale):
            self.conductance = 1e-3*np.exp(log_scale)
            return sum(np.sum(((self.predict(run["t"], run["initial_temperature"])-run["temperature"])/run["sigma"])**2) for run in runs)
        result = minimize_scalar(loss, bounds=(-5., 5.), method="bounded", options={"xatol": 1e-11})
        self.conductance = float(1e-3*np.exp(result.x))
        return self

    def predict(self, t, initial_temperature):
        g = self.conductance
        matrix = np.array([[-g-LINK, LINK], [LINK, -g-LINK]])/(AMOUNT*CP)
        rates, basis = np.linalg.eigh(matrix)
        initial = basis.T@(np.asarray(initial_temperature)-TB)
        return TB+(np.exp(np.asarray(t)[:, None]*rates)*initial)@basis.T
