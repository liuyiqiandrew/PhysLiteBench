import numpy as np


def response(e):
    """Dimensionless long-time heat rate, apart from the mobility.

    Time can be rescaled by ``mobility * stiffness``.  In those units the
    mean heat is ``2*r**2`` and the bulk Legendre transform is the expression
    below.  Since this is *heat*, rather than just the rotational work, the
    unconditioned change in trap energy must also be retained.  Its Gaussian
    endpoint tails truncate the SCGF domain and give the two linear pieces
    implemented here.
    """
    r = float(e['drive_ratio'])
    z = float(e['relative_heat'])
    s = r * r
    q = 2.0 * s * z

    # Interior Legendre transform of
    # psi(lambda) = 1 - sqrt(1 + 4 s lambda (1-lambda)).
    bulk = np.sqrt((1.0 + s) * (q * q + 4.0 * s) / (4.0 * s)) - q / 2.0 - 1.0

    # At lambda=1 the stationary final-position integral ceases to exist.
    # Thus q <= -2s has I(q) = -q.
    if q <= -2.0 * s:
        rate = -q
    # For weak circulation (s < 1/3), the initial-position integral also
    # truncates the positive-q side at lambda_c=(s-1)/(s+1).
    elif s < 1.0 / 3.0:
        q_c = 2.0 * s * (3.0 - s) / (1.0 - 3.0 * s)
        if q >= q_c:
            lambda_c = (s - 1.0) / (s + 1.0)
            psi_c = 4.0 * s / (1.0 + s)
            rate = -lambda_c * q - psi_c
        else:
            rate = bulk
    else:
        rate = bulk

    return float(e['stiffness']) * rate


def predict_at(experiments, mobility):
    return mobility*np.asarray([response(e) for e in experiments])


class Model:
    def __init__(self):
        self.mobility = None

    def fit(self, records):
        # I_i = mobility * response_i with independent, known Gaussian noise.
        # The weighted least-squares estimator is therefore also the MLE.
        bases = np.asarray([response(record['input']) for record in records], dtype=float)
        values = np.asarray([record['value'] for record in records], dtype=float)
        sigmas = np.asarray([record['sigma'] for record in records], dtype=float)
        weights = 1.0 / np.square(sigmas)
        denominator = np.sum(weights * np.square(bases))
        if not np.isfinite(denominator) or denominator <= 0.0:
            raise ValueError('Calibration records must contain informative finite uncertainties.')

        estimate = np.sum(weights * bases * values) / denominator
        # Mobility is known a priori to be in this apparatus range.
        self.mobility = float(np.clip(estimate, 0.8, 1.3))
        return self

    def predict(self, experiments):
        return predict_at(experiments, self.mobility)
