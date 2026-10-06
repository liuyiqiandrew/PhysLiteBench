"""SCGF model and calibration interface for the ring-current apparatus.

For the half-filled weakly asymmetric exclusion process in the stated scaling
limit, the leading current SCGF is

    psi(E, lambda; D) = D * lambda * (lambda + 2 * E) / 4.

The calibration data only leave the common rate prefactor ``D`` unknown.
"""
import numpy as np


def predict_at(experiments, diffusivity):
    """Evaluate the asymptotic SCGF at a sequence of experiment inputs."""
    field = np.asarray([e["field"] for e in experiments], dtype=float)
    bias = np.asarray([e["bias"] for e in experiments], dtype=float)
    return np.asarray(diffusivity * bias * (bias + 2.0 * field) / 4.0,
                      dtype=float)


class Model:
    def __init__(self):
        self.diffusivity = 1.

    def fit(self, records):
        """Fit ``D`` by weighted least squares and return this model.

        Each record is an independent Gaussian observation with standard
        deviation ``sigma``, so the maximum-likelihood estimate is the
        weighted regression through the origin for the known SCGF shape.
        """
        records = list(records)
        if records:
            design = np.asarray([
                r["input"]["bias"] *
                (r["input"]["bias"] + 2.0 * r["input"]["field"]) / 4.0
                for r in records
            ], dtype=float)
            values = np.asarray([r["value"] for r in records], dtype=float)
            sigma = np.asarray([r["sigma"] for r in records], dtype=float)

            weights = 1.0 / np.square(sigma)
            denominator = np.sum(weights * np.square(design))
            if np.isfinite(denominator) and denominator > 0.0:
                estimate = np.sum(weights * design * values) / denominator
                if np.isfinite(estimate):
                    self.diffusivity = float(np.clip(estimate, 0.8, 1.2))

        # Keep the documented parameter invariant even for empty or malformed
        # calibration input.
        if not np.isfinite(self.diffusivity):
            self.diffusivity = 1.0
        self.diffusivity = float(np.clip(self.diffusivity, 0.8, 1.2))
        return self

    def predict(self, experiments):
        return predict_at(experiments, self.diffusivity)
