"""Large-ring current SCGF and calibration interface.

The rates of the process have a common multiplicative factor ``D``.  Thus its
tilted generator, and hence its SCGF, is also proportional to ``D``.  At half
filling the large-ring (weak-field) result is

    psi(E, lambda; D) = D * lambda * (lambda + 2 E) / 4.
"""

import numpy as np


_D_MIN = 0.8
_D_MAX = 1.2


def _design_values(experiments):
    """Return the SCGF at unit diffusivity for an iterable of inputs."""
    # Building the arrays explicitly makes an empty input naturally produce an
    # empty, one-dimensional float array as required by the public API.
    # Materialize once so callers may also pass a generator.
    experiments = list(experiments)
    field = np.asarray([experiment["field"] for experiment in experiments], dtype=float)
    bias = np.asarray([experiment["bias"] for experiment in experiments], dtype=float)
    return bias * (bias + 2.0 * field) / 4.0


def predict_at(experiments, diffusivity):
    """Evaluate the leading large-ring SCGF at the supplied inputs."""
    return float(diffusivity) * _design_values(experiments)


class Model:
    def __init__(self):
        self.diffusivity = 1.0

    def fit(self, records):
        """Fit ``D`` by the Gaussian maximum-likelihood estimate.

        Each calibration uncertainty is a known independent standard
        deviation, so this is a one-parameter inverse-variance weighted least
        squares problem.  The physical public range is imposed on the result.
        """
        records = list(records)
        if not records:
            # There is no information with which to update the initial,
            # physical parameter value.  Keeping it is preferable to creating
            # an undefined 0/0 estimate.
            self.diffusivity = float(np.clip(self.diffusivity, _D_MIN, _D_MAX))
            return self

        unit_prediction = _design_values(record["input"] for record in records)
        values = np.asarray([record["value"] for record in records], dtype=float)
        sigma = np.asarray([record["sigma"] for record in records], dtype=float)

        if (not np.isfinite(unit_prediction).all() or not np.isfinite(values).all()
                or not np.isfinite(sigma).all() or np.any(sigma <= 0.0)):
            raise ValueError("calibration records must contain finite values and positive sigma")

        weights = 1.0 / np.square(sigma)
        denominator = np.sum(weights * np.square(unit_prediction))
        if denominator > 0.0 and np.isfinite(denominator):
            estimate = np.sum(weights * unit_prediction * values) / denominator
            if np.isfinite(estimate):
                self.diffusivity = float(np.clip(estimate, _D_MIN, _D_MAX))
                return self

        # Degenerate records (all have the two fluctuation-theorem zeros) do
        # not identify D.  Preserve a valid existing/default value.
        self.diffusivity = float(np.clip(self.diffusivity, _D_MIN, _D_MAX))
        return self

    def predict(self, experiments):
        return predict_at(experiments, self.diffusivity)
