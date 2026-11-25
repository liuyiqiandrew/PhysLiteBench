import numpy as np


def response(e):
    """Return the rate function with the common mobility factored out.

    For the dimensionless heat ``z = Q/(q_bar*t)``, the stationary
    Ornstein--Uhlenbeck calculation gives

        I(z) = mobility * stiffness * (
            sqrt((1 + r**2) * (1 + r**2*z**2)) - r**2*z - 1
        ).

    The mobility is deliberately omitted here so that it can be fitted once
    from the calibration set and then shared by all preparations.
    """
    r = float(e['drive_ratio'])
    z = float(e['relative_heat'])
    rate = np.sqrt((1.0 + r*r) * (1.0 + r*r*z*z)) - r*r*z - 1.0
    # The exact rate is nonnegative; remove only roundoff-sized negatives at
    # the minimum (z=1), where the exact value is zero.
    rate = max(0.0, float(rate))
    return float(e['stiffness']) * rate


def predict_at(experiments, mobility):
    return mobility*np.asarray([response(e) for e in experiments])


class Model:
    def __init__(self):
        self.mobility = None

    def fit(self, records):
        """Fit the common mobility by weighted least squares.

        Each calibration value has mean ``mobility * response(input)`` and
        known Gaussian standard deviation ``sigma``.  The resulting
        one-parameter maximum-likelihood estimate is the weighted regression
        through the origin, constrained to the apparatus' allowed mobility
        interval.
        """
        if not records:
            raise ValueError('at least one calibration record is required')

        x = np.asarray([response(record['input']) for record in records], dtype=float)
        y = np.asarray([record['value'] for record in records], dtype=float)
        sigma = np.asarray([record['sigma'] for record in records], dtype=float)

        if (not np.isfinite(x).all() or not np.isfinite(y).all()
                or not np.isfinite(sigma).all() or np.any(sigma <= 0)):
            raise ValueError('calibration records must contain finite values and positive sigma')

        weights = 1.0 / (sigma * sigma)
        denominator = np.sum(weights * x * x)
        if denominator == 0.0:
            raise ValueError('calibration records contain no rate information')

        estimate = np.sum(weights * x * y) / denominator
        self.mobility = float(np.clip(estimate, 0.8, 1.3))
        return self

    def predict(self, experiments):
        if self.mobility is None:
            raise RuntimeError('fit must be called before predict')
        return predict_at(experiments, self.mobility)
