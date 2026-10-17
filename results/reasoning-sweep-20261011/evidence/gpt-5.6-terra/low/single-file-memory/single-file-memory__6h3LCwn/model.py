"""Asymptotic tagged-particle covariance for a deterministic single file."""

import numpy as np


def covariance(diffusivity, density, time_a, time_b):
    """Return the quenched long-time covariance coefficient.

    The initially equally-spaced file is a *quenched* initial condition. Its
    two-time covariance therefore contains ``sqrt(t_a + t_b)``; the expression
    with separate square roots of the two times applies to an annealed file.
    """
    diffusivity = float(diffusivity)
    density = float(density)
    time_a = float(time_a)
    time_b = float(time_b)
    if not (np.isfinite(diffusivity) and np.isfinite(density)
            and np.isfinite(time_a) and np.isfinite(time_b)):
        raise ValueError("diffusivity, density, and times must be finite")
    if diffusivity < 0 or density <= 0 or time_a < 0 or time_b < 0:
        raise ValueError("diffusivity and times must be non-negative; density positive")
    return (np.sqrt(diffusivity / np.pi) / density
            * (np.sqrt(time_a + time_b) - np.sqrt(abs(time_a - time_b))))


def predict_at(experiments, diffusivity):
    """Evaluate the documented experiment mapping, preserving input order."""
    return np.asarray([
        covariance(diffusivity, e["density"], e["time_a"], e["time_b"])
        for e in experiments
    ], dtype=float)


class Model:
    def __init__(self):
        self.diffusivity = None

    def fit(self, records):
        """Infer the common diffusivity by weighted Gaussian least squares."""
        records = list(records)
        if not records:
            raise ValueError("at least one calibration record is required")

        factors = np.empty(len(records), dtype=float)
        values = np.empty(len(records), dtype=float)
        sigmas = np.empty(len(records), dtype=float)
        for i, record in enumerate(records):
            try:
                experiment = record["input"]
                values[i] = float(record["value"])
                sigmas[i] = float(record["sigma"])
                factors[i] = covariance(1.0, experiment["density"],
                                        experiment["time_a"], experiment["time_b"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError("invalid calibration record at index %d" % i) from exc

        if (not np.isfinite(values).all() or not np.isfinite(sigmas).all()
                or np.any(sigmas <= 0) or np.any(factors <= 0)):
            raise ValueError("calibration values must be finite, with positive sigma")

        # prediction_i = sqrt(D) * factor_i, so the MLE is a weighted
        # one-parameter linear regression through the origin.
        weights = 1.0 / np.square(sigmas)
        root_d = np.sum(weights * factors * values) / np.sum(weights * np.square(factors))
        root_d = np.clip(root_d, np.sqrt(0.8), np.sqrt(1.2))
        self.diffusivity = float(np.square(root_d))
        return self

    def predict(self, experiments):
        if self.diffusivity is None:
            raise RuntimeError("fit must be called before predict")
        return predict_at(experiments, self.diffusivity)
