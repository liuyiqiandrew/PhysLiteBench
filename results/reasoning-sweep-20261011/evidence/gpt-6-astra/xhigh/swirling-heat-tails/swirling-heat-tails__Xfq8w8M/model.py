"""Long-time heat fluctuations for the stationary circulating trap.

With tau = mobility * stiffness * t and y = sqrt(stiffness) * x,
the mean heat per unit tau is 2*r**2.  Heat is circulating work minus
the change in |y|**2/2; the unconditioned endpoint energies affect its
large deviations even though they do not affect the mean rate.

For E[exp(-lambda*Q)], the interior scaled cumulant generating function
per unit tau is 1 - omega, where
omega = sqrt(1 + 4*r**2*lambda*(1-lambda)).  Stationary endpoint integrals
require omega + 1 > 2*abs(lambda), in addition to a real omega.  Thus
the upper tilt boundary is 1.  The lower boundary is (r**2-1)/(r**2+1)
when r**2 < 1/3, and the lower square-root branch point otherwise.
The Legendre transform over this domain gives the branches below.
"""

import numpy as np


def response(e):
    """Return the heat rate function at unit mobility."""
    r, z = e['drive_ratio'], e['relative_heat']
    s = r * r

    if z < -1.0:
        # The optimal tilt reaches the final-energy boundary lambda = 1.
        rate = -2.0 * s * z
    elif s < 1.0 / 3.0 and z * (1.0 - 3.0 * s) > 3.0 - s:
        # The initial-energy boundary is reached at z = (3-s)/(1-3*s).
        rate = 2.0 * s * ((1.0 - s) * z - 2.0) / (1.0 + s)
    else:
        # Rationalize sqrt((1+s)*(1+s*z*z)) - 1 - s*z to retain
        # precision near the zero at z = 1.
        root = np.sqrt((1.0 + s) * (1.0 + s * z * z))
        rate = s * (z - 1.0)**2 / (root + 1.0 + s * z)

    return e['stiffness'] * rate


def predict_at(experiments, mobility):
    return mobility * np.asarray([response(e) for e in experiments], dtype=float)


class Model:
    def __init__(self):
        self.mobility = None

    def fit(self, records):
        """Fit the shared mobility by bounded Gaussian maximum likelihood."""
        records = list(records)
        values = np.asarray([r['value'] for r in records], dtype=float)
        sigma = np.asarray([r['sigma'] for r in records], dtype=float)
        if not np.all(np.isfinite(values)):
            raise ValueError('Calibration values must be finite.')
        if not np.all(np.isfinite(sigma) & (sigma > 0.0)):
            raise ValueError('Calibration standard deviations must be positive and finite.')

        basis = predict_at([r['input'] for r in records], 1.0) / sigma
        information = np.dot(basis, basis)
        if not np.isfinite(information) or information <= 0.0:
            raise ValueError('Calibration must contain information about mobility.')

        estimate = np.dot(basis, values / sigma) / information
        self.mobility = float(np.clip(estimate, 0.8, 1.3))
        return self

    def predict(self, experiments):
        if self.mobility is None:
            raise RuntimeError('Fit the model before predicting.')
        return predict_at(experiments, self.mobility)
