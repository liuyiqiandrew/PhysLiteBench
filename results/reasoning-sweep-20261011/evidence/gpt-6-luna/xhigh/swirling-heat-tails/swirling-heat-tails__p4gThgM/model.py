import numpy as np


def response(e):
    """Return the dimensionless heat rate function, in stiffness units.

    In time units ``tau = mobility * stiffness * t``, the heat SCGF on its
    finite domain is ``1 - sqrt(1 - 4 r^2 s (1+s))``.  The free stationary
    endpoints restrict that domain: the lower edge is ``s=-1`` and, for
    sufficiently small drive, the upper edge is set by the initial
    stationary distribution.  Their Legendre transform gives the linear
    tails below.
    """
    r = float(e['drive_ratio'])
    z = float(e['relative_heat'])
    k = float(e['stiffness'])
    r2 = r * r
    q = 2.0 * r2 * z

    # The final endpoint creates the lower linear branch at z=-1.
    if q < -2.0 * r2:
        return k * (-q)

    # For r^2 < 1/3 the initial stationary Gaussian cuts off the upper
    # SCGF domain before its square-root singularity, producing another
    # linear branch.  For larger drives the interior expression covers all
    # finite z.
    if r2 < 1.0 / 3.0:
        s_edge = (1.0 - r2) / (1.0 + r2)
        lambda_edge = 4.0 * r2 / (1.0 + r2)
        q_edge = 2.0 * r2 * (3.0 - r2) / (1.0 - 3.0 * r2)
        if q > q_edge:
            return k * (s_edge * q - lambda_edge)

    # Interior Legendre transform.  This rationalized form avoids loss of
    # precision from subtracting nearly equal numbers around the minimum
    # (which is zero at z=1).
    root = np.sqrt((1.0 + r2) * (1.0 + (q / (2.0 * r)) ** 2))
    denom = 4.0 * r2 * (root + 1.0 + q / 2.0)
    rate = (q - 2.0 * r2) ** 2 / denom
    return k * rate


def predict_at(experiments, mobility):
    return mobility*np.asarray([response(e) for e in experiments])


class Model:
    def __init__(self):
        self.mobility = None

    def fit(self, records):
        if not records:
            raise ValueError('At least one calibration record is required.')

        base = np.asarray([response(record['input']) for record in records], dtype=float)
        values = np.asarray([record['value'] for record in records], dtype=float)
        sigma = np.asarray([record['sigma'] for record in records], dtype=float)
        if np.any(~np.isfinite(base)) or np.any(~np.isfinite(values)):
            raise ValueError('Calibration inputs and values must be finite.')
        if np.any(~np.isfinite(sigma)) or np.any(sigma <= 0.0):
            raise ValueError('Calibration standard deviations must be finite and positive.')

        weights = 1.0 / (sigma * sigma)
        denom = np.sum(weights * base * base)
        if denom <= 0.0 or not np.isfinite(denom):
            raise ValueError('Calibration records do not identify a finite mobility.')
        estimate = np.sum(weights * base * values) / denom
        self.mobility = float(np.clip(estimate, 0.8, 1.3))
        return self

    def predict(self, experiments):
        return predict_at(experiments, self.mobility)
