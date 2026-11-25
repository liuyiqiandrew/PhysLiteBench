"""Current SCGF model and calibration interface."""
import numpy as np


def predict_at(experiments, diffusivity):
    """Evaluate the leading large-ring current SCGF.

    At half filling the homogeneous hydrodynamic profile has mobility
    ``2 D rho (1-rho) = D/2``.  Its current SCGF is therefore
    ``D * (bias**2 + 2 * field * bias) / 4``.
    """
    field = np.asarray([e["field"] for e in experiments], dtype=float)
    bias = np.asarray([e["bias"] for e in experiments], dtype=float)
    return float(diffusivity) * bias * (bias + 2.0 * field) / 4.0


class Model:
    def __init__(self):
        self.diffusivity = 1.

    def fit(self, records):
        """Fit the common diffusivity by weighted least squares.

        The model is linear in diffusivity and has no additive offset, so the
        Gaussian maximum-likelihood estimate is a one-parameter weighted
        regression through the origin.  Clipping enforces the documented
        physical parameter interval.
        """
        if not records:
            return self

        field = np.asarray([r["input"]["field"] for r in records], dtype=float)
        bias = np.asarray([r["input"]["bias"] for r in records], dtype=float)
        values = np.asarray([r["value"] for r in records], dtype=float)
        sigma = np.asarray([r["sigma"] for r in records], dtype=float)

        unit_diffusivity_prediction = bias * (bias + 2.0 * field) / 4.0
        weights = 1.0 / np.square(sigma)
        denominator = np.sum(weights * np.square(unit_diffusivity_prediction))
        if denominator > 0.0 and np.isfinite(denominator):
            estimate = np.sum(weights * unit_diffusivity_prediction * values) / denominator
            if np.isfinite(estimate):
                self.diffusivity = float(np.clip(estimate, 0.8, 1.2))
        return self

    def predict(self, experiments):
        return predict_at(experiments, self.diffusivity)
