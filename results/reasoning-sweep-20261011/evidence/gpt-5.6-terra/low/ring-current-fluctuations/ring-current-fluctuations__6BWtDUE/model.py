"""SCGF for the weakly driven, half-filled exclusion ring.

The only fitted quantity is the common microscopic jump-rate prefactor.  In
the weak-asymmetry scaling used in the problem the macroscopic fluctuation
result at density one half is

    psi / D = lambda * (lambda + 2 E) / 4.
"""

import numpy as np


def predict_at(experiments, diffusivity):
    """Return the leading scaled SCGF for an iterable of input dictionaries."""
    # Building the arrays separately deliberately also gives the documented
    # ``(0,)`` result for an empty list.
    field = np.asarray([item["field"] for item in experiments], dtype=float)
    bias = np.asarray([item["bias"] for item in experiments], dtype=float)
    return float(diffusivity) * bias * (bias + 2.0 * field) / 4.0


class Model:
    def __init__(self):
        self.diffusivity = 1.0

    def fit(self, records):
        """Fit ``D`` by inverse-variance weighted least squares.

        Each calibration record observes ``D * x`` with known independent
        Gaussian error, where ``x = lambda(lambda + 2E)/4``.  This has a
        closed-form one-parameter weighted least-squares estimate.
        """
        records = list(records)
        if not records:
            # There is no information to update the prior-safe default.
            self.diffusivity = 1.0
            return self

        inputs = [record["input"] for record in records]
        design = predict_at(inputs, 1.0)
        values = np.asarray([record["value"] for record in records], dtype=float)
        sigma = np.asarray([record["sigma"] for record in records], dtype=float)

        valid = (
            np.isfinite(design)
            & np.isfinite(values)
            & np.isfinite(sigma)
            & (sigma > 0.0)
        )
        if np.any(valid):
            weights = 1.0 / np.square(sigma[valid])
            denominator = np.dot(weights, np.square(design[valid]))
            if denominator > 0.0 and np.isfinite(denominator):
                estimate = np.dot(weights * design[valid], values[valid]) / denominator
                if np.isfinite(estimate):
                    self.diffusivity = float(np.clip(estimate, 0.8, 1.2))
                    return self

        # Keep the required finite, in-range parameter even for degenerate
        # user-supplied calibration data.
        self.diffusivity = 1.0
        return self

    def predict(self, experiments):
        return predict_at(experiments, self.diffusivity)
