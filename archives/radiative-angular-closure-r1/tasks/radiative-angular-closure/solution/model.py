from scipy.optimize import minimize_scalar
import numpy as np
from scipy.linalg import expm


def transport(experiment):
    directions = np.asarray(experiment["directions"], dtype=float)
    initial = np.asarray(experiment["weights"]) * np.asarray(experiment["modulations"])
    generator = -1j * experiment["wavenumber"] * np.diag(directions[:, 0])
    evolved = expm(experiment["time"] * generator) @ initial
    phase = np.exp(1j * experiment["wavenumber"] * experiment["position"])
    return float(np.real(phase * np.sum(evolved)))


class Model:
    def __init__(self):
        self.absorption = .25

    def fit(self, records):
        amplitudes = np.array([transport(r["input"]) for r in records])
        times = np.array([r["input"]["time"] for r in records])
        values = np.array([r["value"] for r in records])
        sigma = np.array([r["sigma"] for r in records])
        def objective(absorption):
            residual = (amplitudes * np.exp(-absorption*times) - values) / sigma
            return float(residual @ residual)
        result = minimize_scalar(objective, bounds=(.12, .45), method="bounded",
                                 options={"xatol": 1e-12})
        self.absorption = float(result.x)
        return self

    def predict(self, experiments):
        return np.asarray([transport(e) * np.exp(-self.absorption*e["time"])
                           for e in experiments], dtype=float)
