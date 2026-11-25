"""SCGF model and calibration interface for the weakly asymmetric ring."""

import numpy as np


def predict_at(experiments, diffusivity):
    """Evaluate the leading large-ring SCGF at a collection of inputs.

    In the scaling in the README, the half-filled ring has

        psi(E, lambda; D) = D * lambda * (lambda + 2 E) / 4.

    The construction through ``np.asarray`` also gives the required one-
    dimensional empty result when no experiments are supplied.
    """
    field = np.asarray([e["field"] for e in experiments], dtype=float)
    bias = np.asarray([e["bias"] for e in experiments], dtype=float)
    return np.asarray(diffusivity * bias * (bias + 2.0 * field) / 4.0,
                      dtype=float)


class Model:
    def __init__(self):
        self.diffusivity = 1.

    def fit(self, records):
        """Fit ``D`` by the Gaussian weighted least-squares estimator.

        Each prediction is linear in D, so the maximum-likelihood estimator
        has a closed form.  The bounds are part of the public model contract;
        clipping also keeps the attribute valid for unusual calibration data.
        """
        coefficient = np.asarray([
            r["input"]["bias"]
            * (r["input"]["bias"] + 2.0 * r["input"]["field"])
            / 4.0
            for r in records
        ], dtype=float)
        value = np.asarray([r["value"] for r in records], dtype=float)
        sigma = np.asarray([r["sigma"] for r in records], dtype=float)

        if coefficient.size == 0:
            # There is no information in an empty calibration set.  Keep the
            # documented, valid default rather than producing NaN.
            self.diffusivity = 1.0
            return self

        weights = 1.0 / np.square(sigma)
        denominator = np.sum(weights * np.square(coefficient))
        if not np.isfinite(denominator) or denominator <= 0.0:
            estimate = 1.0
        else:
            estimate = np.sum(weights * coefficient * value) / denominator

        if not np.isfinite(estimate):
            estimate = 1.0
        self.diffusivity = float(np.clip(estimate, 0.8, 1.2))
        return self

    def predict(self, experiments):
        return predict_at(experiments, self.diffusivity)
