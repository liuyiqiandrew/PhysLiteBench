import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import ndtr


def predict_at(experiments, width):
    d = np.asarray([e['displacement'] for e in experiments])
    h = np.asarray([e['threshold'] for e in experiments])
    theta = np.asarray([e['theta'] for e in experiments])
    phi = np.asarray([e['phi'] for e in experiments])
    accepted = np.asarray([e['observable'] == 'acceptance' for e in experiments])
    plus = ndtr((d-h)/width)
    minus = ndtr((-d-h)/width)
    coherence = np.sqrt(plus*minus)
    joint = ((plus+minus)+np.cos(theta)*(plus-minus))/4
    joint += np.sin(theta)*np.cos(phi)*coherence/2
    return np.where(accepted, (plus+minus)/2, joint)


class Model:
    def __init__(self):
        self.width = None

    def fit(self, records):
        raise NotImplementedError('Fit the common pointer width from calibration records.')

    def predict(self, experiments):
        return predict_at(experiments, self.width)
