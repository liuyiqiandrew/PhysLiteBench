import numpy as np


def response(e):
    """Heat-rate large-deviation rate per unit mobility."""
    r = float(e['drive_ratio'])
    z = float(e['relative_heat'])

    # This is sqrt((1+r^2)(1+r^2 z^2)) - (1+r^2 z).  The
    # rationalized form avoids cancellation around the minimum at z=1.
    root = np.sqrt((1.0 + r*r) * (1.0 + (r*z)*(r*z)))
    numerator = (r*(z - 1.0))**2
    rate = numerator / (root + 1.0 + r*r*z)
    return float(e['stiffness']) * rate


def predict_at(experiments, mobility):
    return mobility*np.asarray([response(e) for e in experiments])


class Model:
    def __init__(self):
        self.mobility = None

    def fit(self, records):
        # Independent Gaussian readings with known variances give a weighted
        # least-squares estimate for the one shared multiplicative parameter.
        basis = np.asarray([response(record['input']) for record in records], dtype=float)
        values = np.asarray([record['value'] for record in records], dtype=float)
        sigma = np.asarray([record['sigma'] for record in records], dtype=float)

        if (basis.size == 0 or not np.all(np.isfinite(basis))
                or not np.all(np.isfinite(values))
                or not np.all(np.isfinite(sigma)) or np.any(sigma <= 0)):
            raise ValueError('Calibration records must contain finite values and positive sigmas.')

        weights = 1.0 / sigma**2
        denominator = np.sum(weights * basis**2)
        if denominator <= 0 or not np.isfinite(denominator):
            raise ValueError('Calibration records do not identify the common mobility.')

        estimate = np.sum(weights * basis * values) / denominator
        self.mobility = float(np.clip(estimate, 0.8, 1.3))
        return self

    def predict(self, experiments):
        return predict_at(experiments, self.mobility)
