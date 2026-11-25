import numpy as np
from scipy.fft import dct, idct
from scipy.optimize import minimize_scalar


class TransportModel:
    def __init__(self):
        self.diffusivity = None

    def fit(self, data):
        def objective(log_d):
            self.diffusivity = np.exp(log_d)
            predicted = np.array([self.predict(data["t"], data["x"], initial)
                                  for initial in data["initial"]])
            return np.sum(((predicted - data["concentration"]) / data["sigma"])**2)
        result = minimize_scalar(objective, bounds=(np.log(1e-11), np.log(1e-7)),
                                 method="bounded", options={"xatol": 1e-11})
        self.diffusivity = float(np.exp(result.x))
        return self

    def predict(self, t, x, initial):
        t, x, initial = np.asarray(t), np.asarray(x), np.asarray(initial)
        mean = np.trapezoid(initial, x, axis=0) / (x[-1] - x[0])
        total = mean[0] + 2 * mean[1]
        if total == 0:
            return np.zeros((len(t), len(x), 2))
        r, s = mean / total
        transport = np.array([[1 + r / 3, -2 * r / 3],
                              [4 * s / 3, 4 * (1 - 2 * s / 3)]])
        rates, vectors = np.linalg.eig(transport)
        coefficients = dct(initial, type=1, axis=0) @ np.linalg.inv(vectors).T
        count = len(x) - 1
        spatial = 4 / (x[1] - x[0])**2 * np.sin(np.arange(count + 1) * np.pi / (2 * count))**2
        decay = self.diffusivity * spatial[:, None] * rates[None, :]
        evolved = coefficients[None, :, :] * np.exp(-t[:, None, None] * decay[None, :, :])
        return idct(evolved @ vectors.T, type=1, axis=1).real
