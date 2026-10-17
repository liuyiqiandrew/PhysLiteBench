"""Asymptotic tagged-particle covariance model."""

import numpy as np


_MIN_DIFFUSIVITY = 0.8
_MAX_DIFFUSIVITY = 1.2


def covariance(diffusivity, density, time_a, time_b):
    """Return the limiting covariance coefficient for one experiment."""
    diffusivity = float(diffusivity)
    density = float(density)
    time_a = float(time_a)
    time_b = float(time_b)
    return (np.sqrt(diffusivity / (2.0 * np.pi)) / density *
            (np.sqrt(time_a) + np.sqrt(time_b) -
             np.sqrt(abs(time_a - time_b))))


def predict_at(experiments, diffusivity):
    """Evaluate experiment dictionaries in their input order."""
    return np.asarray([
        covariance(diffusivity, experiment["density"], experiment["time_a"],
                   experiment["time_b"])
        for experiment in experiments
    ], dtype=float)


class Model:
    def __init__(self):
        self.diffusivity = None

    def fit(self, records):
        """Infer the common diffusivity using the Gaussian-error likelihood.

        If ``a_i = covariance(1, ...)``, then the mean response is
        ``a_i * sqrt(D)``.  Its maximum-likelihood estimate is therefore an
        inverse-variance weighted linear fit for ``sqrt(D)``.
        """
        records = list(records)
        if not records:
            raise ValueError("at least one calibration record is required")

        amplitudes = np.asarray([
            covariance(1.0, record["input"]["density"],
                       record["input"]["time_a"], record["input"]["time_b"])
            for record in records
        ], dtype=float)
        values = np.asarray([record["value"] for record in records], dtype=float)
        sigmas = np.asarray([record["sigma"] for record in records], dtype=float)
        if (not np.isfinite(amplitudes).all() or not np.isfinite(values).all()
                or not np.isfinite(sigmas).all() or np.any(sigmas <= 0)):
            raise ValueError("records must be finite and sigmas must be positive")

        weights = 1.0 / np.square(sigmas)
        denominator = np.sum(weights * np.square(amplitudes))
        if not np.isfinite(denominator) or denominator <= 0:
            raise ValueError("calibration records contain no identifiable signal")

        sqrt_diffusivity = np.sum(weights * amplitudes * values) / denominator
        sqrt_diffusivity = np.clip(sqrt_diffusivity, np.sqrt(_MIN_DIFFUSIVITY),
                                   np.sqrt(_MAX_DIFFUSIVITY))
        self.diffusivity = float(sqrt_diffusivity ** 2)
        return self

    def predict(self, experiments):
        if self.diffusivity is None:
            raise RuntimeError("fit must be called before predict")
        return predict_at(experiments, self.diffusivity)
