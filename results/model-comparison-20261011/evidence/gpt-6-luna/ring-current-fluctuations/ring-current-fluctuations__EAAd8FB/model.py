"""Current SCGF model and calibration interface."""
import numpy as np


def predict_at(experiments, diffusivity):
    """Evaluate the leading weakly asymmetric ring SCGF.

    For the half-filled ring, the homogeneous macroscopic fluctuation
    solution gives ``D * lambda * (lambda + 2 E) / 4``.  ``D`` sets the
    microscopic clock rate, so it multiplies the result without rescaling
    either input.
    """
    field = np.asarray([e["field"] for e in experiments], dtype=float)
    bias = np.asarray([e["bias"] for e in experiments], dtype=float)
    return diffusivity*bias*(bias+2*field)/4


class Model:
    def __init__(self):
        self.diffusivity = 1.

    def fit(self, records):
        """Fit the common diffusivity by weighted least squares.

        The supplied standard deviations are known, independent Gaussian
        errors, so the maximum-likelihood estimate is the weighted linear
        regression coefficient through the origin.  Constrain it to the
        documented physical interval.
        """
        if not records:
            return self

        experiments = [record["input"] for record in records]
        design = np.asarray(
            [e["bias"] * (e["bias"] + 2 * e["field"]) / 4
             for e in experiments],
            dtype=float,
        )
        values = np.asarray([record["value"] for record in records], dtype=float)
        sigma = np.asarray([record["sigma"] for record in records], dtype=float)

        valid = np.isfinite(design) & np.isfinite(values) & np.isfinite(sigma) & (sigma > 0)
        if np.any(valid):
            x = design[valid]
            y = values[valid]
            weights = 1.0 / np.square(sigma[valid])
            denominator = np.sum(weights * np.square(x))
            if denominator > 0 and np.isfinite(denominator):
                estimate = np.sum(weights * x * y) / denominator
                if np.isfinite(estimate):
                    self.diffusivity = float(np.clip(estimate, 0.8, 1.2))
        return self

    def predict(self, experiments):
        return predict_at(experiments, self.diffusivity)
