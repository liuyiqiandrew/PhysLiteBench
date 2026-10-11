import numpy as np


def response(e):
    """Return the dimensionless heat rate for one experiment.

    After scaling time by the relaxation rate ``mobility * stiffness``, the
    large-deviation rate of the heat normalized by its stationary mean is

        sqrt((1 + r**2) * (1 + r**2*z**2)) - 1 - r**2*z.

    The rationalized form below avoids loss of precision near the typical
    value z=1, where the rate is zero.
    """
    r = float(e['drive_ratio'])
    z = float(e['relative_heat'])
    root = np.sqrt((1.0 + r*r) * (1.0 + (r*z)*(r*z)))
    rate = (r*r * (z - 1.0)**2) / (root + 1.0 + r*r*z)
    return float(e['stiffness']) * rate


def predict_at(experiments, mobility):
    return mobility*np.asarray([response(e) for e in experiments])


class Model:
    def __init__(self):
        self.mobility = None

    def fit(self, records):
        # Each reading is y_i = mobility * response(input_i) + Gaussian noise.
        # Thus the weighted least-squares estimate is also the Gaussian MLE.
        design = np.asarray([response(record['input']) for record in records],
                            dtype=float)
        values = np.asarray([record['value'] for record in records], dtype=float)
        sigma = np.asarray([record['sigma'] for record in records], dtype=float)

        weights = 1.0 / (sigma * sigma)
        denominator = np.sum(weights * design * design)
        if not np.isfinite(denominator) or denominator <= 0.0:
            raise ValueError('Calibration records contain no informative inputs.')

        estimate = np.sum(weights * design * values) / denominator
        # The apparatus specifies this physical parameter range.  Constraining
        # the estimate also keeps predictions physical for noisy calibrations.
        self.mobility = float(np.clip(estimate, 0.8, 1.3))
        return self

    def predict(self, experiments):
        if self.mobility is None:
            raise RuntimeError('Model must be fitted before prediction.')
        return predict_at(experiments, self.mobility)
