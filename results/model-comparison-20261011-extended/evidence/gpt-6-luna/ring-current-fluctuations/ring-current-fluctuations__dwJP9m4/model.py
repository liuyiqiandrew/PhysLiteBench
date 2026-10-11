"""Current SCGF model and calibration interface."""

import numpy as np


def predict_at(experiments, diffusivity):
    """Evaluate the leading weak-asymmetry SCGF for a collection of inputs.

    At half filling the homogeneous branch is

        psi(E, lambda; D) = D * lambda * (lambda + 2 E) / 4.

    ``experiments`` is an iterable of dictionaries with ``field`` and ``bias``
    entries.  The array construction also handles an empty iterable.
    """
    experiments = list(experiments)
    if not experiments:
        return np.empty(0, dtype=float)

    field = np.asarray([e["field"] for e in experiments], dtype=float)
    bias = np.asarray([e["bias"] for e in experiments], dtype=float)
    return float(diffusivity) * bias * (bias + 2.0 * field) / 4.0


class Model:
    def __init__(self):
        self.diffusivity = 1.0

    def fit(self, records):
        """Estimate the common diffusivity by weighted least squares."""
        records = list(records)
        if not records:
            self.diffusivity = 1.0
            return self

        inputs = [record["input"] for record in records]
        basis = predict_at(inputs, 1.0)
        values = np.asarray([record["value"] for record in records], dtype=float)
        sigma = np.asarray([record["sigma"] for record in records], dtype=float)

        # The model is linear in D. For independent Gaussian errors, the
        # inverse-variance weighted least-squares estimate has this closed form.
        weights = 1.0 / np.square(sigma)
        denominator = np.sum(weights * np.square(basis))
        if denominator > 0.0 and np.isfinite(denominator):
            estimate = np.sum(weights * basis * values) / denominator
            self.diffusivity = float(np.clip(estimate, 0.8, 1.2))
        else:
            # Degenerate calibration inputs carry no information about D.
            self.diffusivity = 1.0
        return self

    def predict(self, experiments):
        return predict_at(experiments, self.diffusivity)
