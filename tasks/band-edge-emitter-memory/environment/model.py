import numpy as np

BOUNDS = (.59, 1.35)


def _response(experiments, gain):
    result = []
    for e in experiments:
        if e["observable"] != "isolated_transfer":
            raise NotImplementedError
        result.append(np.sin(gain * e["setting"])**2)
    return np.array(result)


class Model:
    def fit(self, records):
        # Set self.gain from the calibration measurements.
        raise NotImplementedError

    def predict(self, experiments):
        return _response(experiments, self.gain)
