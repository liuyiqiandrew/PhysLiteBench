import numpy as np


def covariances(experiment):
    frequency = experiment['frequency']
    interaction = experiment['interaction']
    stiffness = np.array([[1., 2*interaction*np.sqrt(frequency)],
                          [2*interaction*np.sqrt(frequency), frequency**2]])
    values, modes = np.linalg.eigh(stiffness)
    frequencies = np.sqrt(values)
    occupation = 1/np.expm1(frequencies/experiment['temperature'])
    position = (modes*((occupation+.5)/frequencies))@modes.T
    momentum = (modes*((occupation+.5)*frequencies))@modes.T
    ground_position = (modes*(.5/frequencies))@modes.T
    ground_momentum = (modes*(.5*frequencies))@modes.T
    return position, momentum, ground_position, ground_momentum


def response(experiment):
    q, p, q0, p0 = covariances(experiment)
    return float((q[0, 0]+p[0, 0]-q0[0, 0]-p0[0, 0])/2)


class Model:
    def __init__(self):
        self.detection_rate = None

    def fit(self, records):
        raise NotImplementedError("Fit the common detection rate from calibration.")

    def predict(self, experiments):
        return self.detection_rate*np.array([response(e) for e in experiments])
