"""Completed homogeneous conditioned-profile control."""
import numpy as np

def fit_diffusivity(records):
    field = np.asarray([r["input"]["field"] for r in records], dtype=float)
    bias = np.asarray([r["input"]["bias"] for r in records], dtype=float)
    coefficient = bias*(bias+2*field)/4
    values = np.asarray([r["value"] for r in records], dtype=float)
    sigma = np.asarray([r["sigma"] for r in records], dtype=float)
    return float(np.clip(np.sum(coefficient*values/sigma**2)
                         / np.sum(coefficient**2/sigma**2), .8, 1.2))


def predict_at(experiments, diffusivity):
    field = np.asarray([e["field"] for e in experiments], dtype=float)
    bias = np.asarray([e["bias"] for e in experiments], dtype=float)
    return diffusivity*bias*(bias+2*field)/4


class Model:
    def __init__(self):
        self.diffusivity = 1.

    def fit(self, records):
        self.diffusivity = fit_diffusivity(records)
        return self

    def predict(self, experiments):
        return predict_at(experiments, self.diffusivity)
