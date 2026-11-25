"""Long-time heat fluctuations for a stationary circulating OU trap.

In time units tau = mobility * stiffness * t, the mean heat rate is
2*r**2. For E[exp(-lambda*Q)], the bulk scaled cumulant generator is
1 - sqrt(1 + 4*r**2*lambda*(1-lambda)). Stationary, unconditioned
endpoints restrict its domain: lambda < 1, and the lower endpoint is
(r**2-1)/(r**2+1) when r**2 < 1/3, or the lower square-root branch
point otherwise. The restricted Legendre transform gives the heat rate
below, including the linear tails induced by endpoint potential energy.
"""

import numpy as np


def response(e):
    """Return I(z) per unit mobility, in the apparatus time units."""
    a = float(e['drive_ratio']) ** 2
    z = float(e['relative_heat'])
    if z <= -1.0:
        rate = -2.0 * a * z
    elif a < 1.0 / 3.0 and z >= (3.0 - a) / (1.0 - 3.0 * a):
        rate = 2.0 * a * ((1.0 - a) * z - 2.0) / (1.0 + a)
    else:
        # Rationalized bulk expression avoids cancellation near z = 1.
        root = np.sqrt((1.0 + a) * (1.0 + a * z * z))
        rate = a * (z - 1.0) ** 2 / (root + 1.0 + a * z)
    return float(e['stiffness']) * rate


def predict_at(experiments, mobility):
    return mobility * np.asarray([response(e) for e in experiments], dtype=float)


class Model:
    def __init__(self):
        self.mobility = None

    def fit(self, records):
        """Fit the common mobility by bounded Gaussian maximum likelihood."""
        records = list(records)
        x = np.asarray([response(r['input']) for r in records], dtype=float)
        y = np.asarray([r['value'] for r in records], dtype=float)
        sigma = np.asarray([r['sigma'] for r in records], dtype=float)
        if (not records or not np.isfinite(sigma).all() or
                np.any(sigma <= 0) or not np.isfinite(y).all() or
                not np.isfinite(x).all()):
            raise ValueError('Calibration requires finite readings and positive sigmas.')
        xw, yw = x / sigma, y / sigma
        information = np.dot(xw, xw)
        if information == 0:
            raise ValueError('Calibration does not identify mobility.')
        self.mobility = float(np.clip(np.dot(xw, yw) / information, 0.8, 1.3))
        return self

    def predict(self, experiments):
        if self.mobility is None:
            raise ValueError('Fit the model before predicting.')
        return predict_at(experiments, self.mobility)
