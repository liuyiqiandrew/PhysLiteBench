import numpy as np


def response(e):
    """Return the rate with the common mobility factored out.

    For the rotational Ornstein-Uhlenbeck process, the SCGF of the heat
    current is

        mu * stiffness * (1 - sqrt(1 - 4*r**2*lambda*(1+lambda))).

    Legendre transforming it after normalising by the stationary heat rate
    ``2 * mu * stiffness * r**2`` gives the expression below.  The
    rationalised form avoids loss of precision close to the typical value
    ``z = 1`` and is manifestly non-negative.
    """
    r = float(e['drive_ratio'])
    z = float(e['relative_heat'])
    s = r * r
    root = np.sqrt((1.0 + s) * (1.0 + s * z * z))
    rate = s * (z - 1.0) ** 2 / (root + 1.0 + s * z)
    return float(e['stiffness']) * rate


def predict_at(experiments, mobility):
    return float(mobility) * np.asarray([response(e) for e in experiments], dtype=float)


class Model:
    def __init__(self):
        self.mobility = None

    def fit(self, records):
        records = list(records)
        if not records:
            raise ValueError('At least one calibration record is required.')

        # The calibration readings have independent, known Gaussian errors.
        # Since the response is linear in the common mobility, the weighted
        # least-squares estimate has a closed form.
        design = np.asarray([response(record['input']) for record in records], dtype=float)
        values = np.asarray([record['value'] for record in records], dtype=float)
        sigma = np.asarray([record['sigma'] for record in records], dtype=float)
        if not (np.isfinite(design).all() and np.isfinite(values).all() and np.isfinite(sigma).all()):
            raise ValueError('Calibration records must contain finite values.')
        if np.any(sigma <= 0.0):
            raise ValueError('Calibration standard deviations must be positive.')

        weights = 1.0 / (sigma * sigma)
        denominator = np.sum(weights * design * design)
        if denominator <= 0.0:
            raise ValueError('Calibration records contain no informative response.')
        estimate = np.sum(weights * design * values) / denominator
        self.mobility = float(np.clip(estimate, 0.8, 1.3))
        return self

    def predict(self, experiments):
        if self.mobility is None:
            raise ValueError('Model must be fitted before prediction.')
        return predict_at(experiments, self.mobility)
