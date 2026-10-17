import numpy as np


def response(e):
    """Heat rate function with the (unit) mobility factored out.

    The conservative part of the force contributes only the endpoint change
    in trap energy.  The circulating part has mean heat rate
    ``2 * mobility * stiffness * drive_ratio**2``.  Its stationary
    large-deviation rate function, expressed in terms of heat divided by
    that mean rate, is the expression below.
    """
    stiffness = float(e['stiffness'])
    drive_ratio = float(e['drive_ratio'])
    z = float(e['relative_heat'])

    r2 = drive_ratio * drive_ratio
    # This form avoids the removable division by r in an equivalent
    # expression and remains well-conditioned throughout the input domain.
    radical = np.sqrt((1.0 + r2) * (1.0 + r2 * z * z))
    # Rationalizing the difference makes the required I(1) = 0 exact,
    # including for small drive ratios.
    rate = r2 * (z - 1.0) ** 2 / (radical + 1.0 + r2 * z)
    return stiffness * rate


def predict_at(experiments, mobility):
    return mobility*np.asarray([response(e) for e in experiments])


class Model:
    def __init__(self):
        self.mobility = None

    def fit(self, records):
        """Fit the common mobility by Gaussian weighted least squares."""
        records = list(records)
        if not records:
            raise ValueError('at least one calibration record is required')

        unit_rates = np.asarray([response(record['input']) for record in records],
                                dtype=float)
        observations = np.asarray([record['value'] for record in records],
                                  dtype=float)
        sigmas = np.asarray([record['sigma'] for record in records], dtype=float)

        if (not np.isfinite(unit_rates).all() or not np.isfinite(observations).all()
                or not np.isfinite(sigmas).all() or np.any(sigmas <= 0.0)):
            raise ValueError('calibration records must contain finite values and positive sigmas')

        # I_i = mobility * unit_rate_i + independent N(0, sigma_i^2).
        weights = 1.0 / np.square(sigmas)
        denominator = np.sum(weights * np.square(unit_rates))
        if not np.isfinite(denominator) or denominator <= 0.0:
            raise ValueError('calibration data contain no information about mobility')
        estimate = np.sum(weights * unit_rates * observations) / denominator

        # Mobility is a known physical parameter of this apparatus, not an
        # unconstrained regression coefficient.
        self.mobility = float(np.clip(estimate, 0.8, 1.3))
        return self

    def predict(self, experiments):
        return predict_at(experiments, self.mobility)
