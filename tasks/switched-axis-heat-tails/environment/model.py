import numpy as np

BOUNDS = (.57, 1.36)


def _response(experiments, gain):
    result = []
    for e in experiments:
        if e["observable"] != "mean_heat":
            raise NotImplementedError
        stage = e["stages"][0]
        omega = gain * stage["drive"]
        result.append(sum(stage["weights"]) * stage["temperature"][0] * omega**2 / stage["stiffness"][0])
    return np.array(result)


class Model:
    def fit(self, records):
        # Set self.gain from the calibration measurements.
        raise NotImplementedError

    def predict(self, experiments):
        return _response(experiments, self.gain)
