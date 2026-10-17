import numpy as np


def response(e):
    """Rate function per unit ``mobility * stiffness``.

    With time measured in units of ``mobility * stiffness``, the mean heat
    rate is ``2 * r**2``.  The tilted OU eigenvalue gives the usual smooth
    (work) rate function below.  Heat differs from work by the unconditioned
    change in trap energy.  Integrating over the stationary initial and final
    positions restricts the tilt domain: its Legendre transform supplies the
    two linear heat tails implemented here.
    """
    r = float(e['drive_ratio'])
    z = float(e['relative_heat'])
    r2 = r * r
    h = 2.0 * r2 * z  # heat rate in dimensionless time

    # Interior transform of psi(lambda)=1-sqrt(1+4 r^2 lambda(1-lambda)).
    # hypot avoids an unnecessary square before the square root.
    rate = np.sqrt(1.0 + r2) * np.hypot(h, 2.0 * r) / (2.0 * r) - h / 2.0 - 1.0

    # A large negative heat can be produced by an extensive final energy.
    # The final-state normalizability boundary is lambda=1.
    if z <= -1.0:
        rate = -h

    # For weak circulation, a large positive heat can instead be produced by
    # an extensive stationary initial energy.  This boundary occurs before
    # the square-root branch point only when r^2 < 1/3.
    if r2 < 1.0 / 3.0:
        z_critical = (3.0 - r2) / (1.0 - 3.0 * r2)
        if z >= z_critical:
            lambda_min = (r2 - 1.0) / (r2 + 1.0)
            psi_at_min = 4.0 * r2 / (r2 + 1.0)
            rate = -lambda_min * h - psi_at_min

    # The analytic result is non-negative; remove round-off at its zero.
    return float(e['stiffness']) * max(float(rate), 0.0)


def predict_at(experiments, mobility):
    return mobility*np.asarray([response(e) for e in experiments])


class Model:
    def __init__(self):
        self.mobility = None

    def fit(self, records):
        # The rate is exactly linear in the one shared mobility.  Known
        # Gaussian reading errors therefore make this weighted least-squares
        # estimate the maximum-likelihood calibration.
        base = np.asarray([response(record['input']) for record in records], dtype=float)
        values = np.asarray([record['value'] for record in records], dtype=float)
        sigma = np.asarray([record['sigma'] for record in records], dtype=float)
        weights = 1.0 / np.square(sigma)
        estimate = np.sum(weights * base * values) / np.sum(weights * base * base)

        # The apparatus supplies a physical prior bound, so retain it even if
        # a noisy calibration set would put the unconstrained MLE outside it.
        self.mobility = float(np.clip(estimate, 0.8, 1.3))
        return self

    def predict(self, experiments):
        return predict_at(experiments, self.mobility)
