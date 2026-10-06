"""Prediction and calibration model for the ring-current SCGF.

For the half-filled weakly asymmetric exclusion process, the leading
large-ring SCGF in the convention used by the README is

    psi(E, lambda; D) = D * lambda * (lambda + 2 * E) / 4.

The rate prefactor multiplies the whole generator, so it is also a simple
overall scale factor for the observable.  The calibration therefore reduces
to a weighted least-squares fit through the origin.
"""

import numpy as np


_DIFFUSIVITY_BOUNDS = (0.8, 1.2)


def predict_at(experiments, diffusivity):
    """Evaluate the asymptotic SCGF at a sequence of input dictionaries."""
    if not experiments:
        return np.empty(0, dtype=float)

    field = np.asarray([experiment["field"] for experiment in experiments],
                       dtype=float)
    bias = np.asarray([experiment["bias"] for experiment in experiments],
                      dtype=float)
    values = float(diffusivity) * bias * (bias + 2.0 * field) / 4.0
    return np.asarray(values, dtype=float)


class Model:
    def __init__(self):
        self.diffusivity = 1.0

    def fit(self, records):
        """Fit and store the diffusivity, returning this model instance.

        Each observation has a known independent Gaussian standard deviation,
        so inverse-variance weighting gives the maximum-likelihood estimate
        for the one-parameter linear model.
        """
        if not records:
            # There is no information in an empty calibration set.  Keep the
            # documented default, which is already inside the allowed range.
            self.diffusivity = 1.0
            return self

        design = np.asarray([
            record["input"]["bias"] *
            (record["input"]["bias"] + 2.0 * record["input"]["field"]) / 4.0
            for record in records
        ], dtype=float)
        observations = np.asarray([record["value"] for record in records],
                                  dtype=float)
        sigma = np.asarray([record["sigma"] for record in records],
                           dtype=float)

        weights = 1.0 / np.square(sigma)
        denominator = np.sum(weights * np.square(design))
        if not np.isfinite(denominator) or denominator <= 0.0:
            estimate = 1.0
        else:
            estimate = np.sum(weights * design * observations) / denominator

        low, high = _DIFFUSIVITY_BOUNDS
        self.diffusivity = float(np.clip(estimate, low, high))
        return self

    def predict(self, experiments):
        return predict_at(experiments, self.diffusivity)
