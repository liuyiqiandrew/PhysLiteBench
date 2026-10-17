"""Long-time heat large deviations for the stationary circulating trap.

In time units tau = mobility * stiffness * t, the mean heat rate is
2*r**2. For the convention E[exp(-lambda*Q)], the bulk scaled cumulant
function is 1 - eta, eta = sqrt(1 + 4*r**2*lambda*(1-lambda)).
Heat includes U(0)-U(t). Integrating both stationary, unconditioned
endpoints requires eta + 1 > 2*abs(lambda), in addition to eta being real.
The Legendre transform must therefore use this restricted domain: its
upper endpoint is 1; its lower endpoint is (r**2-1)/(r**2+1) when
r**2 < 1/3, and the bulk square-root endpoint otherwise. These restrictions
produce linear tails that are absent from the work rate function.
"""

import numpy as np


def response(e):
    """Heat rate at unit mobility, including stationary endpoint tails."""
    s = float(e['drive_ratio']) ** 2
    z = float(e['relative_heat'])
    q = 2.0 * s * z
    if z <= -1.0:
        rate = -q
    elif s < 1.0 / 3.0 and z >= (3.0 - s) / (1.0 - 3.0 * s):
        rate = ((1.0 - s) * q - 4.0 * s) / (1.0 + s)
    else:
        # Rationalize sqrt((1+s)*(1+s*z*z)) - 1 - s*z to
        # preserve accuracy and nonnegativity near the mean z=1.
        root = np.sqrt((1.0 + s) * (1.0 + s * z * z))
        rate = s * (z - 1.0) ** 2 / (root + 1.0 + s * z)
    return float(e['stiffness']) * rate


def predict_at(experiments, mobility):
    return mobility * np.asarray([response(e) for e in experiments], dtype=float)


class Model:
    def __init__(self):
        self.mobility = None

    def fit(self, records):
        """Fit the common mobility by constrained Gaussian maximum likelihood."""
        records = list(records)
        if not records:
            raise ValueError('Calibration records must not be empty.')
        basis = np.asarray([response(r['input']) for r in records])
        values = np.asarray([r['value'] for r in records], dtype=float)
        sigma = np.asarray([r['sigma'] for r in records], dtype=float)
        if not (np.isfinite(sigma).all() and (sigma > 0).all()
                and np.isfinite(values).all() and np.isfinite(basis).all()):
            raise ValueError('Calibration requires finite values and positive sigma.')
        x, y = basis / sigma, values / sigma
        information = np.dot(x, x)
        if information == 0:
            raise ValueError('Calibration does not identify the mobility.')
        self.mobility = float(np.clip(np.dot(x, y) / information, 0.8, 1.3))
        return self

    def predict(self, experiments):
        if self.mobility is None:
            raise ValueError('Fit the model before predicting.')
        return predict_at(experiments, self.mobility)
