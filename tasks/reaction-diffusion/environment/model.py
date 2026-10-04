import numpy as np
from scipy.fft import dct, idct


class TransportModel:
    def __init__(self):
        self.diffusivity = None

    def fit(self, data):
        raise NotImplementedError

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
