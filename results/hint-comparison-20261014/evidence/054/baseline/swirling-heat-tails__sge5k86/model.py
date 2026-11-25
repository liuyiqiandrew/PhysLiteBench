import numpy as np


def response(e):
    """Rate function apart from the (common) mobility.

    In units of ``mobility * stiffness``, the mean heat rate is
    ``2 * drive_ratio**2``.  Contracting the tilted OU process gives the
    corresponding rate function for heat divided by that mean:

        sqrt((1 + r**2) * (1 + r**2 * z**2)) - 1 - r**2 * z.

    Writing it directly in terms of ``z`` avoids an unnecessary division by
    the drive ratio and is well conditioned throughout the specified domain.
    """
    r = float(e['drive_ratio'])
    z = float(e['relative_heat'])
    stiffness = float(e['stiffness'])
    return stiffness * (np.sqrt((1.0 + r * r) * (1.0 + (r * z) ** 2))
                        - 1.0 - r * r * z)


def predict_at(experiments, mobility):
    return mobility*np.asarray([response(e) for e in experiments])


class Model:
    def __init__(self):
        self.mobility = None

    def fit(self, records):
        """Fit the shared mobility by Gaussian maximum likelihood.

        Each reading has mean ``mobility * response(input)`` and a known,
        potentially different standard deviation.  Thus the one-parameter
        weighted least-squares solution is available analytically.
        """
        factors = np.asarray([response(record['input']) for record in records],
                             dtype=float)
        values = np.asarray([record['value'] for record in records], dtype=float)
        sigmas = np.asarray([record['sigma'] for record in records], dtype=float)

        weights = 1.0 / np.square(sigmas)
        denominator = np.sum(weights * np.square(factors))
        if not np.isfinite(denominator) or denominator <= 0.0:
            raise ValueError('Calibration records do not identify mobility.')

        estimate = np.sum(weights * factors * values) / denominator
        # Mobility is known from the apparatus specification to occupy this
        # interval, so use the constrained Gaussian MLE.
        self.mobility = float(np.clip(estimate, 0.8, 1.3))
        return self

    def predict(self, experiments):
        if self.mobility is None:
            raise RuntimeError('Call fit(records) before predict(experiments).')
        return predict_at(experiments, self.mobility)
