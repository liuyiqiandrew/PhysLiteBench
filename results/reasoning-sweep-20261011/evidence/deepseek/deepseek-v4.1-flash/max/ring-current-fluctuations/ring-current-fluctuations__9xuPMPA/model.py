"""Current SCGF model and calibration interface."""
import numpy as np

def predict_at(experiments, diffusivity):
    field = np.asarray([e["field"] for e in experiments], dtype=float)
    bias = np.asarray([e["bias"] for e in experiments], dtype=float)
    return diffusivity*bias*(bias+2*field)/4


class Model:
    def __init__(self):
        self.diffusivity = 1.

    def fit(self, records):
        raise NotImplementedError("Fit diffusivity from the calibration records.")

    def predict(self, experiments):
        return predict_at(experiments, self.diffusivity)
