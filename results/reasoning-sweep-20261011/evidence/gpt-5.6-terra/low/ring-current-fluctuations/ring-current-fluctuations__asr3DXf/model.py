"""SCGF for the weakly asymmetric exclusion process in the README."""

import numpy as np


def predict_at(experiments, diffusivity):
    """Return the leading, large-ring SCGF for a sequence of experiments.

    At half filling the macroscopic fluctuation result for this normalization is
    ``psi / D = lambda * (lambda + 2 E) / 4``.
    """
    field = np.asarray([experiment["field"] for experiment in experiments], dtype=float)
    bias = np.asarray([experiment["bias"] for experiment in experiments], dtype=float)
    return float(diffusivity) * bias * (bias + 2.0 * field) / 4.0


class Model:
    """Fit the common microscopic diffusivity and predict raw SCGF values."""

    _LOWER = 0.8
    _UPPER = 1.2

    def __init__(self):
        self.diffusivity = 1.0

    def fit(self, records):
        """Fit ``D`` by inverse-variance weighted least squares.

        Each calibration observation has mean ``D * x``, where ``x`` is the
        coefficient used by :func:`predict_at`. The constrained optimum is the
        weighted slope clipped to the documented physical interval.
        """
        if not records:
            self.diffusivity = float(np.clip(self.diffusivity, self._LOWER, self._UPPER))
            return self

        fields = np.asarray([record["input"]["field"] for record in records], dtype=float)
        biases = np.asarray([record["input"]["bias"] for record in records], dtype=float)
        values = np.asarray([record["value"] for record in records], dtype=float)
        sigmas = np.asarray([record["sigma"] for record in records], dtype=float)
        x = biases * (biases + 2.0 * fields) / 4.0

        valid = (np.isfinite(x) & np.isfinite(values) & np.isfinite(sigmas)
                 & (sigmas > 0.0))
        weights = np.zeros_like(sigmas)
        weights[valid] = 1.0 / sigmas[valid] ** 2
        denominator = np.sum(weights * x * x)
        if np.isfinite(denominator) and denominator > 0.0:
            estimate = np.sum(weights * x * values) / denominator
            if np.isfinite(estimate):
                self.diffusivity = float(np.clip(estimate, self._LOWER, self._UPPER))
        else:
            self.diffusivity = float(np.clip(self.diffusivity, self._LOWER, self._UPPER))
        return self

    def predict(self, experiments):
        return predict_at(experiments, self.diffusivity)
