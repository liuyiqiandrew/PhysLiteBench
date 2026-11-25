"""Author oracle integrating the coupled heat/work capacity matrix."""
import numpy as np
from scipy.integrate import solve_ivp
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
            g = 1e-3*np.exp(log_scale)
            total = 0.
            for run in runs:
                initial = np.asarray(run["initial_temperature"])
                difference = (initial[0]-initial[1])/2*np.exp(-(g+2*LINK)*np.asarray(run["t"])/(AMOUNT*CP))
                value = TB+np.column_stack((difference, -difference))
                total += np.sum(((value-run["temperature"])/run["sigma"])**2)
            return total
        result = minimize_scalar(loss, bounds=(-5., 5.), method="bounded", options={"xatol": 1e-11})
        self.conductance = float(1e-3*np.exp(result.x))
        return self

    def predict(self, t, initial_temperature):
        def rhs(_, temperature):
            transfer = LINK*(temperature[0]-temperature[1])
            heat = -self.conductance*(temperature-TB)+np.array([-transfer, transfer])
            capacity = AMOUNT*CP*np.eye(2)-AMOUNT*R*np.outer(temperature/temperature.sum(), np.ones(2))
            return np.linalg.solve(capacity, heat)
        times = np.asarray(t)
        if times[-1] == 0.:
            return np.tile(initial_temperature, (len(times), 1))
        return solve_ivp(rhs, (0., times[-1]), initial_temperature, t_eval=times,
                         method="DOP853", rtol=1e-11, atol=1e-10).y.T
