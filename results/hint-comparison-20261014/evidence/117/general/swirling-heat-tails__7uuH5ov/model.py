import numpy as np


def response(e):
    """Return the rate with the common mobility factored out.

    In units of the relaxation time ``(mobility * stiffness)**-1``, the
    heat large-deviation function is

        J(z) = sqrt((1+r^2) * (q^2 + 4*r^2) / (4*r^2)) - q/2 - 1,
        q = 2*r^2*z.

    The difference of the square root and the last two terms suffers from
    cancellation around the typical value z=1.  Rationalising that
    difference gives an algebraically identical, non-negative expression.
    """
    r = float(e['drive_ratio'])
    z = float(e['relative_heat'])

    q = 2.0 * r * r * z
    root = np.sqrt((1.0 + r * r) * (q * q + 4.0 * r * r) / (4.0 * r * r))
    numerator = (q - 2.0 * r * r) ** 2 / (4.0 * r * r)
    rate = numerator / (root + 1.0 + q / 2.0)
    return float(e['stiffness']) * rate


def predict_at(experiments, mobility):
    return mobility*np.asarray([response(e) for e in experiments])


class Model:
    def __init__(self):
        self.mobility = None

    def fit(self, records):
        # Every observation has the form value = mobility * response(input)
        # plus independent Gaussian noise.  Thus the maximum-likelihood
        # estimate is the weighted one-parameter least-squares solution.
        x = np.asarray([response(record['input']) for record in records], dtype=float)
        y = np.asarray([record['value'] for record in records], dtype=float)
        sigma = np.asarray([record['sigma'] for record in records], dtype=float)

        weights = 1.0 / (sigma * sigma)
        denominator = np.sum(weights * x * x)
        if not np.isfinite(denominator) or denominator <= 0.0:
            raise ValueError('Calibration records do not identify mobility.')

        mobility = np.sum(weights * x * y) / denominator
        # Enforce the physical range stated by the apparatus.
        self.mobility = float(np.clip(mobility, 0.8, 1.3))
        return self

    def predict(self, experiments):
        return predict_at(experiments, self.mobility)
