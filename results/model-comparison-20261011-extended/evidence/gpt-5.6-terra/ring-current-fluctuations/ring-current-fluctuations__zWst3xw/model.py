"""SCGF for the weakly asymmetric exclusion process in the README.

The field and counting tilt are both of order ``1 / L`` at the microscopic
level.  At density one half the leading large-ring current SCGF is therefore

    D / 4 * lambda * (lambda + 2 E).

``D`` is the only fitted parameter; the data values have independent normal
errors, so its maximum-likelihood estimate is a weighted least-squares slope.
"""

import numpy as np


_LOWER_DIFFUSIVITY = 0.8
_UPPER_DIFFUSIVITY = 1.2


def _design_values(experiments):
    """Return the coefficient multiplying diffusivity for each input."""
    field = np.asarray([experiment["field"] for experiment in experiments], dtype=float)
    bias = np.asarray([experiment["bias"] for experiment in experiments], dtype=float)
    return bias * (bias + 2.0 * field) / 4.0


def predict_at(experiments, diffusivity):
    """Predict raw (not standardized) SCGF values for an iterable of inputs."""
    coefficient = _design_values(experiments)
    result = float(diffusivity) * coefficient
    # This also makes the empty-input result explicitly one-dimensional.
    return np.asarray(result, dtype=float)


class Model:
    def __init__(self):
        self.diffusivity = 1.0

    def fit(self, records):
        """Fit diffusivity by inverse-variance weighted least squares.

        For observations ``y_i = D*x_i + noise_i`` with known standard
        deviations ``sigma_i``, the unconstrained MLE is
        ``sum(x_i*y_i/sigma_i**2) / sum(x_i**2/sigma_i**2)``.  The required
        parameter interval is imposed by projection of that scalar estimate.
        """
        records = list(records)
        if not records:
            # A data-free fit is well-defined as retaining the model's current
            # valid parameter, which is useful to callers assembling batches.
            self.diffusivity = float(np.clip(self.diffusivity,
                                              _LOWER_DIFFUSIVITY,
                                              _UPPER_DIFFUSIVITY))
            return self

        inputs = [record["input"] for record in records]
        x = _design_values(inputs)
        y = np.asarray([record["value"] for record in records], dtype=float)
        sigma = np.asarray([record["sigma"] for record in records], dtype=float)

        if (not np.isfinite(x).all() or not np.isfinite(y).all() or
                not np.isfinite(sigma).all() or np.any(sigma <= 0.0)):
            raise ValueError("calibration values and positive sigmas must be finite")

        weights = 1.0 / np.square(sigma)
        denominator = float(np.dot(weights, np.square(x)))
        if denominator > 0.0 and np.isfinite(denominator):
            estimate = float(np.dot(weights * x, y) / denominator)
        else:
            # Measurements at x=0 contain no information about diffusivity.
            estimate = self.diffusivity

        self.diffusivity = float(np.clip(estimate,
                                         _LOWER_DIFFUSIVITY,
                                         _UPPER_DIFFUSIVITY))
        return self

    def predict(self, experiments):
        return predict_at(experiments, self.diffusivity)
