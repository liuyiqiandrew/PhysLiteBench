"""Large-deviation model for the heat dissipated by the circulating trap."""

import numpy as np


def response(experiment):
    """Return the dimensionless heat rate function (per ``mobility*k``).

    With ``a = drive_ratio**2``, time can be measured in units of
    ``mobility * stiffness`` and the mean heat current is ``2*a``.  The
    usual tilted-OU eigenvalue gives the curved part of the rate function.
    Heat is work minus the change in trap energy, however, so the stationary
    initial and final energy distributions restrict the counting-field
    domain.  Legendre transforming on that restricted domain produces the
    two possible linear tails below.
    """
    r = float(experiment['drive_ratio'])
    z = float(experiment['relative_heat'])
    a = r * r

    # The final-energy endpoint restricts the counting field to lambda < 1.
    # Thus z < -1 is a linear (rather than work-like) heat tail.
    if z < -1.0:
        return -2.0 * a * z

    # For a < 1/3 the initial-energy endpoint imposes an additional lower
    # counting-field boundary.  Its tangent takes over above this z value.
    if a < (1.0 / 3.0):
        z_endpoint = (3.0 - a) / (1.0 - 3.0 * a)
        if z > z_endpoint:
            return 2.0 * a * ((1.0 - a) * z - 2.0) / (1.0 + a)

    # Curved branch, written in terms of z to avoid cancellation from the
    # intermediate current q = 2*a*z.
    return np.sqrt((1.0 + a) * (1.0 + a * z * z)) - a * z - 1.0


def predict_at(experiments, mobility):
    """Predict physical inverse-time rates for an iterable of inputs."""
    rates = np.asarray(
        [response(e) * float(e['stiffness']) for e in experiments], dtype=float
    )
    return float(mobility) * rates


class Model:
    def __init__(self):
        self.mobility = None

    def fit(self, records):
        """Fit the common mobility using the known Gaussian uncertainties."""
        if not records:
            raise ValueError('at least one calibration record is required')

        unit_rates = np.asarray(
            [response(record['input']) * float(record['input']['stiffness'])
             for record in records],
            dtype=float,
        )
        values = np.asarray([record['value'] for record in records], dtype=float)
        sigma = np.asarray([record['sigma'] for record in records], dtype=float)
        if (not np.isfinite(unit_rates).all() or not np.isfinite(values).all()
                or not np.isfinite(sigma).all() or np.any(sigma <= 0.0)):
            raise ValueError('calibration records must contain finite positive sigmas')

        # Weighted zero-intercept Gaussian maximum-likelihood estimate.
        weights = 1.0 / (sigma * sigma)
        denominator = np.sum(weights * unit_rates * unit_rates)
        if denominator <= 0.0 or not np.isfinite(denominator):
            raise ValueError('calibration inputs do not identify mobility')
        estimate = np.sum(weights * unit_rates * values) / denominator

        # The apparatus specifies the physically admissible interval.
        self.mobility = float(np.clip(estimate, 0.8, 1.3))
        return self

    def predict(self, experiments):
        if self.mobility is None:
            raise RuntimeError('fit must be called before predict')
        return predict_at(experiments, self.mobility)
