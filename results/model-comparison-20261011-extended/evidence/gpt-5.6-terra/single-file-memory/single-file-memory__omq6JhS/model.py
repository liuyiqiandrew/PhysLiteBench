"""Asymptotic tagged-particle covariance model.

For a file of Brownian point particles initially placed on a lattice, the
large-time covariance of the tagged particle is proportional to ``sqrt(D)``.
This makes inference of the one common diffusivity a one-parameter weighted
linear least-squares problem after using ``sqrt(D)`` as the parameter.
"""

import math
from collections.abc import Mapping

import numpy as np


_MIN_DIFFUSIVITY = 0.8
_MAX_DIFFUSIVITY = 1.2


def covariance(diffusivity, density, time_a, time_b):
    """Return the limiting covariance coefficient for one experiment."""
    diffusivity = float(diffusivity)
    density = float(density)
    time_a = float(time_a)
    time_b = float(time_b)

    values = (diffusivity, density, time_a, time_b)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("diffusivity, density, and times must be finite")
    if diffusivity < 0 or density <= 0 or time_a < 0 or time_b < 0:
        raise ValueError("diffusivity and times must be non-negative and density positive")

    # This is symmetric in the two observation times.  At equal times it
    # reduces to sqrt(2*D*t/pi)/density.
    time_factor = math.sqrt(time_a) + math.sqrt(time_b) - math.sqrt(abs(time_a - time_b))
    return math.sqrt(diffusivity / (2.0 * math.pi)) * time_factor / density


def _experiment_factor(experiment):
    """Covariance at unit diffusivity, with a helpful input error on failure."""
    if not isinstance(experiment, Mapping):
        raise TypeError("each experiment must be a mapping")
    try:
        return covariance(1.0, experiment["density"], experiment["time_a"], experiment["time_b"])
    except KeyError as exc:
        raise ValueError(f"experiment is missing required field {exc.args[0]!r}") from exc


def predict_at(experiments, diffusivity):
    """Predict covariance coefficients in the same order as ``experiments``."""
    if diffusivity is None:
        raise RuntimeError("fit must be called before predicting")
    diffusivity = float(diffusivity)
    if not math.isfinite(diffusivity) or diffusivity < 0:
        raise ValueError("diffusivity must be a finite non-negative number")

    # np.fromiter gives a guaranteed one-dimensional float array, including
    # for an empty generator of experiments.
    return np.fromiter(
        (math.sqrt(diffusivity) * _experiment_factor(experiment) for experiment in experiments),
        dtype=float,
    )


class Model:
    def __init__(self):
        self.diffusivity = None

    def fit(self, records):
        """Fit the common diffusivity by weighted Gaussian maximum likelihood.

        With ``f_i = covariance(1, ...)`` and ``q = sqrt(D)``, each record
        has mean ``q*f_i``. Its known Gaussian sigma therefore gives the
        closed-form weighted least-squares estimate used below.
        """
        numerator = 0.0
        denominator = 0.0
        count = 0

        for record in records:
            if not isinstance(record, Mapping):
                raise TypeError("each calibration record must be a mapping")
            try:
                experiment = record["input"]
                observed = float(record["value"])
                sigma = float(record["sigma"])
            except KeyError as exc:
                raise ValueError(f"calibration record is missing required field {exc.args[0]!r}") from exc

            if not math.isfinite(observed) or not math.isfinite(sigma) or sigma <= 0:
                raise ValueError("calibration values must be finite and sigma must be positive")

            factor = _experiment_factor(experiment)
            weight = 1.0 / (sigma * sigma)
            numerator += weight * factor * observed
            denominator += weight * factor * factor
            count += 1

        if count == 0 or not math.isfinite(denominator) or denominator <= 0:
            raise ValueError("at least one informative calibration record is required")

        # The apparatus specifies a known interval for D. Projecting the
        # unconstrained MLE onto the corresponding interval for sqrt(D) is
        # the constrained Gaussian MLE.
        sqrt_diffusivity = numerator / denominator
        sqrt_diffusivity = min(
            math.sqrt(_MAX_DIFFUSIVITY),
            max(math.sqrt(_MIN_DIFFUSIVITY), sqrt_diffusivity),
        )
        self.diffusivity = sqrt_diffusivity * sqrt_diffusivity
        return self

    def predict(self, experiments):
        return predict_at(experiments, self.diffusivity)
