"""Long-time heat-fluctuation rate for the circulating harmonic trap.

The dimensionless part of the rate only depends on the drive ratio.  The
overall clock rate is ``mobility * stiffness``.
"""

import numpy as np


def response(experiment):
    """Return the rate with mobility and stiffness factored apart.

    For a drive ratio ``r``, set ``a = r**2``.  In rescaled time the mean
    heat rate is ``2*a``, so ``relative_heat`` is the reported normalized
    heat.  The bulk Legendre transform is supplemented by the endpoint
    (internal-energy) tails appropriate to heat rather than work.
    """
    r = float(experiment["drive_ratio"])
    z = float(experiment["relative_heat"])
    a = r * r

    # The branch obtained from the tilted OU generator in its normalizable
    # range.  It applies between the endpoint-induced linear tails.
    bulk = np.sqrt((1.0 + a) * (1.0 + a * z * z)) - a * z - 1.0

    # Integrating over the final stationary endpoint restricts the tilt to
    # lambda < 1.  Its Legendre transform is the negative-heat tail.
    if z < -1.0:
        rate = -2.0 * a * z
    # For weak circulation, integrating the initial stationary endpoint
    # also caps the negative tilt.  The positive-heat tail starts here.
    elif a < (1.0 / 3.0) and z > (3.0 - a) / (1.0 - 3.0 * a):
        rate = (2.0 * a / (1.0 + a)) * ((1.0 - a) * z - 2.0)
    else:
        rate = bulk

    return float(experiment["stiffness"]) * rate


def predict_at(experiments, mobility):
    """Predict rates for a sequence of public input dictionaries."""
    return float(mobility) * np.asarray([response(e) for e in experiments], dtype=float)


class Model:
    def __init__(self):
        self.mobility = None

    def fit(self, records):
        """Fit the common mobility by constrained Gaussian least squares."""
        records = list(records)
        if not records:
            raise ValueError("at least one calibration record is required")

        base = np.asarray([response(record["input"]) for record in records], dtype=float)
        observed = np.asarray([record["value"] for record in records], dtype=float)
        sigma = np.asarray([record["sigma"] for record in records], dtype=float)
        if (
            not np.isfinite(base).all()
            or not np.isfinite(observed).all()
            or not np.isfinite(sigma).all()
            or np.any(sigma <= 0.0)
        ):
            raise ValueError("calibration records must have finite values and positive sigma")

        weights = 1.0 / np.square(sigma)
        denominator = np.sum(weights * np.square(base))
        if denominator == 0.0:
            raise ValueError("calibration data contain no mobility information")

        unconstrained = np.sum(weights * base * observed) / denominator
        # This is the maximum-likelihood estimate with the apparatus' known
        # physical mobility bounds.
        self.mobility = float(np.clip(unconstrained, 0.8, 1.3))
        return self

    def predict(self, experiments):
        if self.mobility is None:
            raise RuntimeError("fit must be called before predict")
        return predict_at(experiments, self.mobility)
