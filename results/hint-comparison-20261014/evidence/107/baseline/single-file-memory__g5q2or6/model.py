"""Asymptotic tagged-particle covariance model.

For the fixed (rather than equilibrium-random) initial particle lattice, the
single-file covariance is proportional to ``sqrt(D)``.  Fitting the one
unknown material parameter is therefore a weighted linear least-squares fit.
"""

import numpy as np


_MIN_DIFFUSIVITY = 0.8
_MAX_DIFFUSIVITY = 1.2


def covariance(diffusivity, density, time_a, time_b):
    """Return the long-time covariance coefficient for one experiment."""
    diffusivity = float(diffusivity)
    density = float(density)
    time_a = float(time_a)
    time_b = float(time_b)
    return (np.sqrt(diffusivity / (2.0 * np.pi)) / density
            * (np.sqrt(time_a) + np.sqrt(time_b)
               - np.sqrt(abs(time_a - time_b))))


def _unit_diffusivity_coefficient(experiment):
    """Coefficient multiplying ``sqrt(diffusivity)`` in an experiment."""
    try:
        density = float(experiment["density"])
        time_a = float(experiment["time_a"])
        time_b = float(experiment["time_b"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("each experiment needs numeric density, time_a, and time_b") from exc

    if (not np.isfinite(density) or not np.isfinite(time_a)
            or not np.isfinite(time_b) or density <= 0.0
            or time_a < 0.0 or time_b < 0.0):
        raise ValueError("density must be positive and times must be non-negative finite values")

    return ((np.sqrt(time_a) + np.sqrt(time_b) - np.sqrt(abs(time_a - time_b)))
            / (density * np.sqrt(2.0 * np.pi)))


def predict_at(experiments, diffusivity):
    """Predict covariance coefficients, preserving the experiments' order."""
    if diffusivity is None:
        raise RuntimeError("fit must be called before predict")
    diffusivity = float(diffusivity)
    if not np.isfinite(diffusivity) or diffusivity < 0.0:
        raise ValueError("diffusivity must be a finite non-negative value")

    coefficients = np.asarray(
        [_unit_diffusivity_coefficient(experiment) for experiment in experiments],
        dtype=float,
    )
    return coefficients * np.sqrt(diffusivity)


class Model:
    def __init__(self):
        self.diffusivity = None

    def fit(self, records):
        """Fit the shared diffusivity by inverse-variance weighted least squares.

        With ``c_i`` as the unit-diffusivity prediction, a datum obeys
        ``value_i = c_i * sqrt(D) + error_i``.  The Gaussian maximum-likelihood
        estimate is consequently the closed-form weighted slope below.
        """
        records = list(records)
        if not records:
            raise ValueError("at least one calibration record is required")

        coefficients = []
        values = []
        sigmas = []
        for record in records:
            try:
                coefficient = _unit_diffusivity_coefficient(record["input"])
                value = float(record["value"])
                sigma = float(record["sigma"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError("each record needs input, value, and sigma") from exc
            if not np.isfinite(value) or not np.isfinite(sigma) or sigma <= 0.0:
                raise ValueError("record values must be finite and sigma must be positive")
            coefficients.append(coefficient)
            values.append(value)
            sigmas.append(sigma)

        coefficients = np.asarray(coefficients, dtype=float)
        values = np.asarray(values, dtype=float)
        weights = 1.0 / np.square(np.asarray(sigmas, dtype=float))
        denominator = np.sum(weights * np.square(coefficients))
        if not np.isfinite(denominator) or denominator <= 0.0:
            raise ValueError("calibration records contain no informative experiments")

        sqrt_diffusivity = np.sum(weights * coefficients * values) / denominator
        sqrt_diffusivity = np.clip(
            sqrt_diffusivity,
            np.sqrt(_MIN_DIFFUSIVITY),
            np.sqrt(_MAX_DIFFUSIVITY),
        )
        self.diffusivity = float(sqrt_diffusivity * sqrt_diffusivity)
        return self

    def predict(self, experiments):
        return predict_at(experiments, self.diffusivity)
