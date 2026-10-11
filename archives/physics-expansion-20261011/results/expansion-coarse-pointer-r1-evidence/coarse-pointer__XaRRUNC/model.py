import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import ndtr


def predict_at(experiments, width):
    """Return the ideal readout probabilities for a given pointer width."""
    d = np.asarray([e['displacement'] for e in experiments])
    h = np.asarray([e['threshold'] for e in experiments])
    theta = np.asarray([e['theta'] for e in experiments])
    phi = np.asarray([e['phi'] for e in experiments])
    accepted = np.asarray([e['observable'] == 'acceptance' for e in experiments])

    # The two sigma_z components translate the pointer to +d and -d.
    plus = ndtr((d - h) / width)
    minus = ndtr((-d - h) / width)

    # This is the overlap of the two translated wavefunctions restricted to
    # q >= h.  It is not, in general, sqrt(plus * minus): the latter is only
    # an approximation to a truncated-wavefunction overlap.
    coherence = np.exp(-d * d / (2 * width * width)) * ndtr(-h / width)

    joint = ((plus+minus)+np.cos(theta)*(plus-minus))/4
    joint += np.sin(theta)*np.cos(phi)*coherence/2
    return np.where(accepted, (plus+minus)/2, joint)


class Model:
    def __init__(self):
        self.width = None

    def fit(self, records):
        """Fit the common width by weighted least squares on calibration data."""
        experiments = [record['input'] for record in records]
        values = np.asarray([record['value'] for record in records], dtype=float)
        sigmas = np.asarray([record['sigma'] for record in records], dtype=float)

        def objective(width):
            residual = (predict_at(experiments, width) - values) / sigmas
            return np.sum(residual * residual)

        result = minimize_scalar(
            objective,
            bounds=(0.6, 1.0),
            method='bounded',
            options={'xatol': 1e-12},
        )
        self.width = float(result.x)
        return self

    def predict(self, experiments):
        return predict_at(experiments, self.width)
