import numpy as np

BOUNDS = (.60, 1.34)


def _response(experiments, gain):
    result = []
    for e in experiments:
        if e["observable"] != "energy_density":
            raise NotImplementedError
        result.append(-e["inverse_temperature"] * gain**2 * sum(e["variances"]))
    return np.array(result)


class Model:
    def fit(self, records):
        # Set self.gain from the calibration measurements.
        raise NotImplementedError

    def predict(self, experiments):
        return _response(experiments, self.gain)
