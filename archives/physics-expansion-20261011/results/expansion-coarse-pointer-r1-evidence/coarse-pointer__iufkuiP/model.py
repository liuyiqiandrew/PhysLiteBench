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
    # The interference contribution is the overlap of the two pointer
    # wavefunctions over the accepted part of the position record.  It is
    # not, in general, sqrt(plus * minus), since the latter treats the
    # coarse-grained pointer states as if they were normalized Gaussians.
    coherence = np.exp(-d**2/(2*width**2))*ndtr(-h/width)
    joint = ((plus+minus)+np.cos(theta)*(plus-minus))/4
    joint += np.sin(theta)*np.cos(phi)*coherence/2
    return np.where(accepted, (plus+minus)/2, joint)


class Model:
    def __init__(self):
        self.width = None

    def fit(self, records):
        values = np.asarray([r['value'] for r in records], dtype=float)
        sigmas = np.asarray([r['sigma'] for r in records], dtype=float)
        experiments = [r['input'] for r in records]

        def objective(width):
            residual = (predict_at(experiments, width) - values) / sigmas
            return np.sum(residual**2)

        result = minimize_scalar(objective, bounds=(0.6, 1.0), method='bounded')
        self.width = float(result.x)
        return self

    def predict(self, experiments):
        return predict_at(experiments, self.width)
