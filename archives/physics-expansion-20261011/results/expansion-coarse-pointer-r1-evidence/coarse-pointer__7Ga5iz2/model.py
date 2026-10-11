import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import ndtr


def predict_at(experiments, width):
    """Return the acceptance or joint probability for each experiment.

    The two pointer components after the interaction are Gaussians centered
    at +d and -d.  The interference contribution has to be integrated over
    the accepted half-line; it is therefore the overlap of those Gaussians
    on q >= h, rather than the geometric mean of their separate tail
    probabilities.
    """
    d = np.asarray([e['displacement'] for e in experiments], dtype=float)
    h = np.asarray([e['threshold'] for e in experiments], dtype=float)
    theta = np.asarray([e['theta'] for e in experiments], dtype=float)
    phi = np.asarray([e['phi'] for e in experiments], dtype=float)
    accepted = np.asarray([e['observable'] == 'acceptance' for e in experiments])

    plus = ndtr((d - h) / width)
    minus = ndtr((-d - h) / width)
    acceptance = (plus + minus) / 2.0

    # Integral_h^infinity psi(q-d) psi(q+d) dq for the pointer state in the
    # README.  The full overlap is exp(-d^2/(2 w^2)); truncating at h adds
    # the factor Phi(-h/w).
    overlap = np.exp(-d * d / (2.0 * width * width)) * ndtr(-h / width)
    joint = ((plus + minus) + np.cos(theta) * (plus - minus)) / 4.0
    joint += np.sin(theta) * np.cos(phi) * overlap / 2.0
    return np.where(accepted, acceptance, joint)


class Model:
    def __init__(self):
        self.width = None

    def fit(self, records):
        records = list(records)
        if not records:
            raise ValueError('At least one calibration record is required.')

        experiments = [record['input'] for record in records]
        values = np.asarray([record['value'] for record in records], dtype=float)
        sigmas = np.asarray([record['sigma'] for record in records], dtype=float)
        if (not np.all(np.isfinite(values)) or
                not np.all(np.isfinite(sigmas)) or np.any(sigmas <= 0.0)):
            raise ValueError('Calibration values and sigmas must be finite, with sigma > 0.')

        def objective(width):
            residual = (predict_at(experiments, width) - values) / sigmas
            return float(np.dot(residual, residual))

        result = minimize_scalar(
            objective,
            bounds=(0.6, 1.0),
            method='bounded',
            options={'xatol': 1e-12},
        )
        if not result.success or not np.isfinite(result.x):
            raise RuntimeError('Unable to fit the common pointer width.')
        self.width = float(np.clip(result.x, 0.6, 1.0))
        return self

    def predict(self, experiments):
        return predict_at(experiments, self.width)
