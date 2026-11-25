"""SCGF model and calibration interface for the ring-current apparatus.

In the weakly asymmetric, half-filled large-ring limit, the leading SCGF is

    psi(E, lambda; D) = D * lambda * (lambda + 2 E) / 4.

The diffusivity is the only unknown parameter, so the calibration is a
weighted least-squares fit through the origin in this known basis function.
"""

import numpy as np


_DIFFUSIVITY_BOUNDS = (0.8, 1.2)


def predict_at(experiments, diffusivity):
    """Evaluate the asymptotic SCGF for a collection of inputs."""
    # Constructing both arrays explicitly also ensures that an empty input
    # produces a one-dimensional, empty NumPy array rather than a scalar.
    field = np.asarray([e["field"] for e in experiments], dtype=float)
    bias = np.asarray([e["bias"] for e in experiments], dtype=float)
    return np.asarray(diffusivity * bias * (bias + 2.0 * field) / 4.0,
                      dtype=float)


class Model:
    def __init__(self):
        self.diffusivity = 1.

    def fit(self, records):
        """Fit ``D`` from records containing input, value, and sigma fields.

        The measurement errors are independent Gaussians with known standard
        deviations, hence inverse-variance weighting is the appropriate
        likelihood fit.  The physical parameter range is enforced after the
        fit so the public model invariant holds even for unusual calibration
        data.
        """
        records = list(records)
        if not records:
            self.diffusivity = 1.0
            return self

        basis = np.asarray([
            r["input"]["bias"] *
            (r["input"]["bias"] + 2.0 * r["input"]["field"]) / 4.0
            for r in records
        ], dtype=float)
        values = np.asarray([r["value"] for r in records], dtype=float)
        sigma = np.asarray([r["sigma"] for r in records], dtype=float)

        if (not np.isfinite(basis).all() or
                not np.isfinite(values).all() or
                not np.isfinite(sigma).all() or
                np.any(sigma <= 0.0)):
            raise ValueError("calibration values and sigmas must be finite; sigmas must be positive")

        weights = 1.0 / np.square(sigma)
        denominator = np.sum(weights * np.square(basis))
        if denominator == 0.0:
            # No record constrains D (all records are at a zero-SCGF input).
            estimate = 1.0
        else:
            estimate = np.sum(weights * basis * values) / denominator

        self.diffusivity = float(np.clip(estimate, *_DIFFUSIVITY_BOUNDS))
        return self

    def predict(self, experiments):
        return predict_at(experiments, self.diffusivity)
