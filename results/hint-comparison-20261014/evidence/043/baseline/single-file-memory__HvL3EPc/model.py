"""Asymptotic covariance model for a tagged particle in a single file."""

import numpy as np


def covariance(diffusivity, density, time_a, time_b):
    """Return the long-time covariance coefficient for one experiment.

    For the initially equally spaced file the tagged-particle process has the
    fractional-Brownian covariance shape

        sqrt(D / (2 pi)) / rho * (sqrt(s) + sqrt(t) - sqrt(|s-t|)).
    """
    diffusivity = float(diffusivity)
    density = float(density)
    time_a = float(time_a)
    time_b = float(time_b)
    return (np.sqrt(diffusivity / (2.0 * np.pi)) / density
            * (np.sqrt(time_a) + np.sqrt(time_b)
               - np.sqrt(abs(time_a - time_b))))


def predict_at(experiments, diffusivity):
    """Evaluate the model for an iterable of documented experiment mappings."""
    return np.asarray([
        covariance(diffusivity, experiment["density"],
                   experiment["time_a"], experiment["time_b"])
        for experiment in experiments
    ], dtype=float)


class Model:
    """Fit and evaluate the common diffusivity of the apparatus."""

    def __init__(self):
        self.diffusivity = None

    def fit(self, records):
        """Infer the common diffusivity from noisy calibration records.

        The covariance is linear in ``theta = sqrt(D)``.  Consequently the
        Gaussian maximum-likelihood estimate is a one-parameter weighted
        least-squares fit, with each supplied ``sigma`` used as its known
        measurement standard deviation.
        """
        records = list(records)
        if not records:
            raise ValueError("at least one calibration record is required")

        amplitudes = []
        values = []
        weights = []
        for record in records:
            experiment = record["input"]
            density = float(experiment["density"])
            time_a = float(experiment["time_a"])
            time_b = float(experiment["time_b"])
            value = float(record["value"])
            sigma = float(record["sigma"])

            if not (np.isfinite(density) and density > 0.0
                    and np.isfinite(time_a) and time_a >= 0.0
                    and np.isfinite(time_b) and time_b >= 0.0
                    and np.isfinite(value) and np.isfinite(sigma)
                    and sigma > 0.0):
                raise ValueError("calibration records must contain finite physical values")

            # covariance(D, ...) = amplitude * sqrt(D)
            amplitude = ((np.sqrt(time_a) + np.sqrt(time_b)
                          - np.sqrt(abs(time_a - time_b)))
                         / (density * np.sqrt(2.0 * np.pi)))
            amplitudes.append(amplitude)
            values.append(value)
            weights.append(1.0 / sigma ** 2)

        amplitudes = np.asarray(amplitudes, dtype=float)
        values = np.asarray(values, dtype=float)
        weights = np.asarray(weights, dtype=float)
        denominator = np.sum(weights * amplitudes ** 2)
        if not np.isfinite(denominator) or denominator <= 0.0:
            raise ValueError("calibration records do not identify diffusivity")

        sqrt_diffusivity = np.sum(weights * amplitudes * values) / denominator
        # The physical range is part of the apparatus specification.  In this
        # one-dimensional convex problem clipping is the constrained MLE.
        sqrt_diffusivity = np.clip(sqrt_diffusivity, np.sqrt(0.8), np.sqrt(1.2))
        self.diffusivity = float(sqrt_diffusivity ** 2)
        return self

    def predict(self, experiments):
        """Return covariance coefficients in the same order as experiments."""
        if self.diffusivity is None:
            raise RuntimeError("fit must be called before predict")
        predictions = predict_at(experiments, self.diffusivity)
        if predictions.ndim != 1 or not np.isfinite(predictions).all():
            raise ValueError("experiments produced non-finite predictions")
        return predictions
