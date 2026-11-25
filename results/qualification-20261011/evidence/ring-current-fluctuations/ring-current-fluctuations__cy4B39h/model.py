"""SCGF model and calibration interface for the ring process.

In the weak-field, half-filled large-ring limit in the question, the leading
SCGF is

    psi(E, lambda; D) = D * lambda * (lambda + 2 * E) / 4.

The calibration therefore only has one unknown scale, ``D``.
"""
import numpy as np


def predict_at(experiments, diffusivity):
    """Evaluate the asymptotic SCGF at a sequence of input dictionaries."""
    field = np.asarray([e["field"] for e in experiments], dtype=float)
    bias = np.asarray([e["bias"] for e in experiments], dtype=float)
    return np.asarray(diffusivity * bias * (bias + 2.0 * field) / 4.0,
                      dtype=float)


class Model:
    def __init__(self):
        self.diffusivity = 1.

    def fit(self, records):
        """Fit ``D`` by weighted least squares and return this model.

        Each calibration value has independent Gaussian error with the
        supplied standard deviation, so the maximum-likelihood estimate for
        the one-parameter model is a weighted regression through the origin.
        """
        basis = np.asarray([
            r["input"]["bias"] *
            (r["input"]["bias"] + 2.0 * r["input"]["field"]) / 4.0
            for r in records
        ], dtype=float)
        values = np.asarray([r["value"] for r in records], dtype=float)
        sigma = np.asarray([r["sigma"] for r in records], dtype=float)

        if basis.size == 0:
            raise ValueError("at least one calibration record is required")
        if (not np.isfinite(basis).all() or
                not np.isfinite(values).all() or
                not np.isfinite(sigma).all() or
                np.any(sigma <= 0.0)):
            raise ValueError("calibration records must contain finite values and positive sigma")

        weights = 1.0 / np.square(sigma)
        denominator = np.sum(weights * np.square(basis))
        if denominator == 0.0 or not np.isfinite(denominator):
            raise ValueError("calibration inputs do not identify diffusivity")

        estimate = np.sum(weights * basis * values) / denominator
        # Keep the public parameter within its documented domain even if an
        # unusual calibration set would place the unconstrained estimate
        # outside it.
        self.diffusivity = float(np.clip(estimate, 0.8, 1.2))
        return self

    def predict(self, experiments):
        return predict_at(experiments, self.diffusivity)
