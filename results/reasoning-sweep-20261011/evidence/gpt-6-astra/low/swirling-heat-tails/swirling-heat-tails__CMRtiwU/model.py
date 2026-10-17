"""Long-time heat fluctuations for a stationary circulating OU trap.

In time units tau = mobility * stiffness * t, the mean heat rate is
2*r**2.  For the Laplace tilt exp(-lambda*Q), the bulk eigenvalue is
psi(lambda) = 1 - sqrt(1 + 4*r**2*lambda*(1-lambda)).

Heat is circulating work minus the change in potential energy. Integrating
both stationary endpoints requires D + 1 > 2*abs(lambda), where D is the
square root above, in addition to the bulk spectral domain. The upper
endpoint is lambda=1. For r**2 < 1/3 the lower endpoint is
(r**2-1)/(r**2+1); otherwise it is the lower spectral branch point.
The restricted Legendre transform gives the branches below.
"""

import numpy as np


def response(e):
    """Reported heat rate at unit mobility, including endpoint tails."""
    a = float(e['drive_ratio']) ** 2
    z = float(e['relative_heat'])
    q = 2.0 * a * z

    if z <= -1.0:
        rate = -q
    elif a < 1.0 / 3.0 and z >= (3.0 - a) / (1.0 - 3.0 * a):
        rate = ((1.0 - a) * q - 4.0 * a) / (1.0 + a)
    else:
        # Rationalize sqrt((1+a)*(1+a*z*z)) - (1+a*z)
        # to avoid cancellation near the zero at z=1.
        root = np.sqrt((1.0 + a) * (1.0 + a * z * z))
        offset = 1.0 + a * z
        if offset >= 0.0:
            rate = a * (z - 1.0) ** 2 / (root + offset)
        else:
            rate = root - offset
    return float(e['stiffness']) * rate


def predict_at(experiments, mobility):
    return mobility * np.asarray([response(e) for e in experiments], dtype=float)


class Model:
    def __init__(self):
        self.mobility = None

    def fit(self, records):
        """Fit the common mobility by bounded Gaussian maximum likelihood."""
        records = list(records)
        if not records:
            raise ValueError('At least one calibration record is required.')
        basis = np.asarray([response(r['input']) for r in records])
        values = np.asarray([r['value'] for r in records], dtype=float)
        sigma = np.asarray([r['sigma'] for r in records], dtype=float)
        if not (np.isfinite(sigma).all() and (sigma > 0.0).all()
                and np.isfinite(values).all() and np.isfinite(basis).all()):
            raise ValueError('Calibration values must be finite and sigmas positive.')
        weighted_basis = basis / sigma
        information = np.dot(weighted_basis, weighted_basis)
        if information == 0.0:
            raise ValueError('Calibration does not identify the mobility.')
        estimate = np.dot(weighted_basis, values / sigma) / information
        self.mobility = float(np.clip(estimate, 0.8, 1.3))
        return self

    def predict(self, experiments):
        if self.mobility is None:
            raise ValueError('Fit the model before predicting.')
        return predict_at(experiments, self.mobility)
