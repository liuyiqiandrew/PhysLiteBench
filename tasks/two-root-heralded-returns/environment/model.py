import numpy as np


class Model:
    def fit(self, records):
        # Fit self.gain to the calibration records.
        raise NotImplementedError

    def predict(self, experiments):
        if not experiments:
            return np.array([], dtype=float)
        raise NotImplementedError
