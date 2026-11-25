import numpy as np


def response(e):
    """Return the rate with the mobility factor omitted.

    For the OU process in the apparatus, the heat SCGF gives

        I(z) = mobility * stiffness *
               (sqrt((1+r^2)(1+r^2 z^2)) - 1 - r^2 z).

    The rationalized form below is algebraically identical, but avoids
    cancellation near the typical value z=1.
    """
    r = float(e['drive_ratio'])
    z = float(e['relative_heat'])
    stiffness = float(e['stiffness'])

    root = np.sqrt((1.0 + r * r) * (1.0 + (r * z) ** 2))
    dimensionless_rate = (r * r * (z - 1.0) ** 2) / (
        root + 1.0 + r * r * z
    )
    return stiffness * dimensionless_rate


def predict_at(experiments, mobility):
    return mobility*np.asarray([response(e) for e in experiments])


class Model:
    def __init__(self):
        self.mobility = None

    def fit(self, records):
        """Fit the common mobility by the Gaussian maximum-likelihood rule."""
        if not records:
            raise ValueError('at least one calibration record is required')

        design = np.asarray([response(record['input']) for record in records],
                            dtype=float)
        values = np.asarray([record['value'] for record in records], dtype=float)
        sigma = np.asarray([record['sigma'] for record in records], dtype=float)

        if (not np.isfinite(design).all() or not np.isfinite(values).all()
                or not np.isfinite(sigma).all() or np.any(sigma <= 0.0)):
            raise ValueError('calibration records must contain finite values')

        weights = 1.0 / (sigma * sigma)
        denominator = np.sum(weights * design * design)
        if denominator == 0.0:
            raise ValueError('calibration records contain no informative inputs')

        # The response is linear in the common mobility and has zero
        # intercept.  Enforce the physical mobility interval from the README.
        estimate = np.sum(weights * design * values) / denominator
        self.mobility = float(np.clip(estimate, 0.8, 1.3))
        return self

    def predict(self, experiments):
        if self.mobility is None:
            raise RuntimeError('fit must be called before predict')
        return predict_at(experiments, self.mobility)
