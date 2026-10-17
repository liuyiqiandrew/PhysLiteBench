"""SCGF model and calibration interface for the ring-current apparatus.

For the half-filled weakly asymmetric ring in the question, the leading
large-ring scaled cumulant generating function is

    psi(E, lambda; D) = D * lambda * (lambda + 2 * E) / 4.

The only unknown quantity is therefore the common time-scale factor D.
"""

import numpy as np


_LOWER_D = 0.8
_UPPER_D = 1.2


def predict_at(experiments, diffusivity):
    """Evaluate the asymptotic SCGF at a sequence of input dictionaries."""
    if not experiments:
        return np.empty(0, dtype=float)

    field = np.asarray([item["field"] for item in experiments], dtype=float)
    bias = np.asarray([item["bias"] for item in experiments], dtype=float)
    return np.asarray(diffusivity * bias * (bias + 2.0 * field) / 4.0,
                      dtype=float)


class Model:
    def __init__(self):
        self.diffusivity = 1.0

    def fit(self, records):
        """Fit the common diffusivity and return this model.

        The calibration errors are independent Gaussians with known standard
        deviations, so the maximum-likelihood estimate is weighted least
        squares through the origin in the variable
        ``bias * (bias + 2 * field) / 4``.
        """
        factors = []
        values = []
        sigmas = []

        for record in records:
            inputs = record["input"]
            field = float(inputs["field"])
            bias = float(inputs["bias"])
            value = float(record["value"])
            sigma = float(record["sigma"])

            if not (np.isfinite(field) and np.isfinite(bias) and
                    np.isfinite(value) and np.isfinite(sigma)):
                continue
            if sigma <= 0.0:
                continue

            factors.append(bias * (bias + 2.0 * field) / 4.0)
            values.append(value)
            sigmas.append(sigma)

        if factors:
            x = np.asarray(factors, dtype=float)
            y = np.asarray(values, dtype=float)
            weights = 1.0 / np.square(np.asarray(sigmas, dtype=float))
            denominator = np.sum(weights * x * x)
            if denominator > 0.0 and np.isfinite(denominator):
                estimate = np.sum(weights * x * y) / denominator
                if np.isfinite(estimate):
                    self.diffusivity = float(np.clip(
                        estimate, _LOWER_D, _UPPER_D))

        # Keep the public parameter valid even if fit receives no usable
        # information (for example, only inputs with zero SCGF factor).
        if not np.isfinite(self.diffusivity):
            self.diffusivity = 1.0
        self.diffusivity = float(np.clip(
            self.diffusivity, _LOWER_D, _UPPER_D))
        return self

    def predict(self, experiments):
        return predict_at(experiments, self.diffusivity)
