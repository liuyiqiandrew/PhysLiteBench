"""Model for the large-ring SCGF of the weakly asymmetric exclusion process.

For the rates in the README, the weak field is ``E/L`` and the counting
field for the extensive jump count is ``lambda/L``.  At half filling, the
leading large-ring eigenvalue is therefore

    psi(E, lambda; D) = D * lambda * (lambda + 2*E) / 4.

The dependence on the unknown rate prefactor is linear, which also makes the
weighted Gaussian calibration a one-parameter least-squares problem.
"""

import numpy as np


_LOWER_DIFFUSIVITY = 0.8
_UPPER_DIFFUSIVITY = 1.2


def _features(experiments):
    """Return the coefficient multiplying diffusivity for each input."""
    if not experiments:
        return np.empty(0, dtype=float)

    field = np.asarray([item["field"] for item in experiments], dtype=float)
    bias = np.asarray([item["bias"] for item in experiments], dtype=float)
    return bias * (bias + 2.0 * field) / 4.0


def predict_at(experiments, diffusivity):
    """Evaluate the asymptotic SCGF at a fixed diffusivity."""
    return np.asarray(diffusivity * _features(experiments), dtype=float)


class Model:
    def __init__(self):
        self.diffusivity = 1.0

    def fit(self, records):
        """Fit and return this model using the supplied Gaussian observations."""
        if not records:
            # Keep the documented parameter valid when no calibration data are
            # supplied; the default is inside the allowed physical interval.
            self.diffusivity = 1.0
            return self

        inputs = [record["input"] for record in records]
        coefficient = _features(inputs)
        values = np.asarray([record["value"] for record in records], dtype=float)
        sigma = np.asarray([record["sigma"] for record in records], dtype=float)

        # Independent Gaussian errors with known standard deviations imply
        # inverse-variance weighting.  The fallback handles a degenerate
        # calibration set without allowing a non-finite parameter.
        valid = np.isfinite(coefficient) & np.isfinite(values) & np.isfinite(sigma)
        valid &= sigma > 0.0
        coefficient = coefficient[valid]
        values = values[valid]
        sigma = sigma[valid]
        weights = 1.0 / np.square(sigma)
        denominator = np.sum(weights * np.square(coefficient))
        if denominator > 0.0 and np.isfinite(denominator):
            estimate = np.sum(weights * coefficient * values) / denominator
        else:
            estimate = 1.0

        # Calibration is expected to be in the public domain.  Clipping keeps
        # the API guarantee intact even for unusually noisy/out-of-domain data.
        self.diffusivity = float(np.clip(estimate, _LOWER_DIFFUSIVITY, _UPPER_DIFFUSIVITY))
        return self

    def predict(self, experiments):
        return predict_at(experiments, self.diffusivity)
