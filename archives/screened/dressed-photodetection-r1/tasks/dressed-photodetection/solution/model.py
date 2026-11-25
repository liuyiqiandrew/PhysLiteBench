import numpy as np


def response(experiment):
    frequency = experiment['frequency']
    interaction = experiment['interaction']
    stiffness = np.array([[1., 2*interaction*np.sqrt(frequency)],
                          [2*interaction*np.sqrt(frequency), frequency**2]])
    values, modes = np.linalg.eigh(stiffness)
    frequencies = np.sqrt(values)
    occupation = 1/np.expm1(frequencies/experiment['temperature'])
    return float(np.sum(modes[0]**2*occupation/frequencies))


class Model:
    def __init__(self):
        self.detection_rate = None

    def fit(self, records):
        x = np.array([response(r['input'])/r['sigma'] for r in records])
        y = np.array([r['value']/r['sigma'] for r in records])
        self.detection_rate = float(x@y/(x@x))
        return self

    def predict(self, experiments):
        return self.detection_rate*np.array([response(e) for e in experiments])
