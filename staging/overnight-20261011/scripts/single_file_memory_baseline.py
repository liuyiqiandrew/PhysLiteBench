import numpy as np


def covariance(diffusivity, density, time_a, time_b):
    return np.sqrt(diffusivity/(2*np.pi))/density * (np.sqrt(time_a)+np.sqrt(time_b)-np.sqrt(abs(time_a-time_b)))


def predict_at(experiments, diffusivity):
    return np.asarray([
        covariance(float(diffusivity), float(e["density"]),
                   float(e["time_a"]), float(e["time_b"]))
        for e in experiments
    ], dtype=float)


class Model:
    def __init__(self):
        self.diffusivity = None

    def fit(self, records):
        records = list(records)
        design = predict_at([r["input"] for r in records], 1.0)
        values = np.asarray([r["value"] for r in records])
        sigma = np.asarray([r["sigma"] for r in records])
        amplitude = np.sum(design*values/sigma**2)/np.sum(design**2/sigma**2)
        self.diffusivity = float(np.clip(amplitude, np.sqrt(.8), np.sqrt(1.2))**2)
        return self

    def predict(self, experiments):
        return predict_at(experiments, self.diffusivity)
