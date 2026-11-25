import numpy as np
from scipy.linalg import solve_continuous_lyapunov


def stationary_covariance(experiment, friction):
    m = float(experiment["mass"])
    k = float(experiment["spring"])
    h = float(experiment["coupling"])
    b = float(experiment["field"])
    temperatures = np.asarray(experiment["temperatures"], dtype=float)
    stiffness = np.array([[k, h], [h, k]])
    rotation = np.array([[0., 1.], [-1., 0.]])
    drift = np.block([[np.zeros((2, 2)), np.eye(2)],
                      [-stiffness / m, (-friction * np.eye(2) + b * rotation) / m]])
    noise = np.zeros((4, 4))
    noise[2:, 2:] = 2 * friction * np.diag(temperatures) / m**2
    covariance = solve_continuous_lyapunov(drift, -noise)
    return (covariance + covariance.T) / 2


def response(experiment, friction):
    covariance = stationary_covariance(experiment, friction)
    k, h = experiment["spring"], experiment["coupling"]
    stiffness = np.array([[k, h], [h, k]])
    powers = np.diag(stiffness @ covariance[:2, 2:])
    return float(powers[int(experiment["bath"])])


class Model:
    def __init__(self):
        self.friction = 1.0

    def fit(self, records):
        raise NotImplementedError("Implement calibration fitting.")

    def predict(self, experiments):
        cache = {}
        out = []
        for experiment in experiments:
            key = (experiment["mass"], experiment["spring"], experiment["coupling"],
                   experiment["field"], *experiment["temperatures"], experiment["bath"])
            if key not in cache:
                cache[key] = response(experiment, self.friction)
            out.append(cache[key])
        return np.asarray(out, dtype=float)
