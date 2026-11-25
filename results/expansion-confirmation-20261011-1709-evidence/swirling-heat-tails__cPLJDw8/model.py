import numpy as np


def response(e):
    """Return the rate function with the common mobility factored out.

    For this rotational Ornstein-Uhlenbeck process, the stationary heat rate
    is ``q_bar = 2 * mobility * stiffness * drive_ratio**2``.  Solving the
    tilted-generator eigenvalue problem and Legendre transforming it gives
    the following rate for ``Q / (q_bar * t)``.  The mobility only sets the
    overall inverse-time scale, so it is deliberately not included here.
    """
    stiffness = float(e['stiffness'])
    drive_ratio = float(e['drive_ratio'])
    relative_heat = float(e['relative_heat'])

    # q is the heat rate in units of mobility * stiffness.  Keeping this
    # intermediate dimensionless quantity makes the fluctuation symmetry
    # I(z) - I(-z) = -2 * stiffness * mobility * r**2 * z explicit.
    q = 2.0 * drive_ratio**2 * relative_heat
    rate = (
        np.sqrt(
            (1.0 + drive_ratio**2)
            * (q**2 + 4.0 * drive_ratio**2)
            / (4.0 * drive_ratio**2)
        )
        - 0.5 * q
        - 1.0
    )
    # Roundoff can make the exact minimum at z=1 appear very slightly
    # negative; the large-deviation rate itself is non-negative.
    return stiffness * max(float(rate), 0.0)


def predict_at(experiments, mobility):
    return mobility*np.asarray([response(e) for e in experiments])


class Model:
    def __init__(self):
        self.mobility = None

    def fit(self, records):
        """Fit the common mobility using the independent Gaussian readings."""
        if not records:
            raise ValueError('at least one calibration record is required')

        design = np.asarray([response(record['input']) for record in records], dtype=float)
        values = np.asarray([record['value'] for record in records], dtype=float)
        sigma = np.asarray([record['sigma'] for record in records], dtype=float)

        if (
            design.ndim != 1
            or values.shape != design.shape
            or sigma.shape != design.shape
            or not np.isfinite(design).all()
            or not np.isfinite(values).all()
            or not np.isfinite(sigma).all()
            or np.any(sigma <= 0.0)
            or np.all(design == 0.0)
        ):
            raise ValueError('calibration records must contain finite, weighted observations')

        # Weighted least squares for y_i = mobility * response_i + noise_i.
        weights = 1.0 / sigma**2
        mobility = np.sum(weights * design * values) / np.sum(weights * design**2)
        self.mobility = float(np.clip(mobility, 0.8, 1.3))
        return self

    def predict(self, experiments):
        return predict_at(experiments, self.mobility)
