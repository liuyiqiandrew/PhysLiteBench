import numpy as np


def response(e):
    """Rate function per unit mobility for one preparation.

    The conservative portion of the heat is only a bounded change in trap
    energy, so it does not contribute to the long-time rate function.  The
    circulating force gives the dimensionless heat rate function below.  Its
    minimum is at ``z == 1``, since the heat has been normalized by its own
    stationary mean rate.
    """
    r = float(e['drive_ratio'])
    z = float(e['relative_heat'])
    stiffness = float(e['stiffness'])
    r2 = r * r
    return stiffness * (np.sqrt((1.0 + r2) * (1.0 + r2 * z * z)) - r2 * z - 1.0)


def predict_at(experiments, mobility):
    return mobility*np.asarray([response(e) for e in experiments])


class Model:
    def __init__(self):
        self.mobility = None

    def fit(self, records):
        """Fit the single common scale using the readings' known variances."""
        base_rates = np.asarray([response(record['input']) for record in records], dtype=float)
        values = np.asarray([record['value'] for record in records], dtype=float)
        sigmas = np.asarray([record['sigma'] for record in records], dtype=float)

        # Independent Gaussian readings: this is the weighted least-squares
        # (and maximum-likelihood) estimate for I = mobility * base_rate.
        weights = 1.0 / np.square(sigmas)
        denominator = np.sum(weights * np.square(base_rates))
        if not np.isfinite(denominator) or denominator <= 0.0:
            raise ValueError('calibration data cannot identify mobility')
        estimate = np.sum(weights * base_rates * values) / denominator
        # Mobility is specified by the apparatus to live in this interval.
        self.mobility = float(np.clip(estimate, 0.8, 1.3))
        return self

    def predict(self, experiments):
        return predict_at(experiments, self.mobility)
