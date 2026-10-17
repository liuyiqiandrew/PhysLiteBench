"""Long-time stationary heat fluctuations in the overdamped circulating trap.

In time units tau = mobility * stiffness * t, the stationary mean heat rate
is 2*r**2.  Heat is circulation work minus the change in trap energy, so the
unconditioned endpoint energies affect its large deviations even as t grows.

For the transform E[exp(-lambda*Q)], the interior scaled cumulant generator
is 1 - omega, where omega = sqrt(1 + 4*r**2*lambda*(1-lambda)).  Integrating
over both stationary endpoints requires 1 + omega > 2*abs(lambda), as well
as real omega.  The rate is the Legendre transform over this restricted
domain, not over the full domain of the circulation-work generator.
"""

import numpy as np


def response(e):
    """Return the heat rate for unit mobility, in apparatus time units."""
    r, z = e["drive_ratio"], e["relative_heat"]
    s = r * r

    if z <= -1.0:
        # The final-energy overlap limits the tilt to lambda < 1.
        rate = -2.0 * s * z
    elif s < 1.0 / 3.0 and z >= (3.0 - s) / (1.0 - 3.0 * s):
        # For weak drive, the initial-energy overlap also limits the tilt:
        # lambda > (s - 1)/(s + 1).  Otherwise the lower spectral endpoint
        # is steep and produces no finite positive-tail transition.
        rate = 2.0 * s * ((1.0 - s) * z - 2.0) / (1.0 + s)
    else:
        # Rationalized sqrt((1+s)*(1+s*z*z)) - 1 - s*z avoids
        # cancellation near the mean z=1 and gives exactly zero there.
        root = np.sqrt((1.0 + s) * (1.0 + s * z * z))
        rate = s * (z - 1.0) ** 2 / (root + 1.0 + s * z)

    return e["stiffness"] * rate


def predict_at(experiments, mobility):
    return mobility * np.asarray([response(e) for e in experiments], dtype=float)


class Model:
    def __init__(self):
        self.mobility = None

    def fit(self, records):
        """Fit the common mobility by bounded Gaussian maximum likelihood."""
        records = list(records)
        if not records:
            raise ValueError("At least one calibration record is required.")

        values = np.asarray([r["value"] for r in records], dtype=float)
        sigmas = np.asarray([r["sigma"] for r in records], dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("Calibration values must be finite.")
        if not np.isfinite(sigmas).all() or np.any(sigmas <= 0.0):
            raise ValueError("Calibration sigmas must be finite and positive.")

        basis = predict_at([r["input"] for r in records], 1.0) / sigmas
        information = np.dot(basis, basis)
        if information == 0.0:
            raise ValueError("Calibration must include rates away from z=1.")

        estimate = np.dot(basis, values / sigmas) / information
        self.mobility = float(np.clip(estimate, 0.8, 1.3))
        return self

    def predict(self, experiments):
        if self.mobility is None:
            raise RuntimeError("Fit the model before predicting rates.")
        return predict_at(experiments, self.mobility)
