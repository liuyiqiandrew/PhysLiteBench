"""Large-ring SCGF model and calibration interface."""

import numpy as np


_DIFFUSIVITY_BOUNDS = (0.8, 1.2)


def _counting_features(experiments):
    """Return the diffusivity-independent part of the asymptotic SCGF.

    At half filling, the weakly asymmetric ring has the leading large-ring
    scaled cumulant generating function

        psi(E, lambda; D) = D * lambda * (lambda + 2 E) / 4.

    The quadratic dependence is the result after taking the long-time limit
    before the large-ring limit, with the counting field per jump equal to
    lambda / L.
    """
    field = np.asarray([item["field"] for item in experiments], dtype=float)
    bias = np.asarray([item["bias"] for item in experiments], dtype=float)
    return bias * (bias + 2.0 * field) / 4.0


def predict_at(experiments, diffusivity):
    """Predict raw asymptotic SCGF values for a sequence of inputs."""
    feature = _counting_features(experiments)
    return np.asarray(diffusivity * feature, dtype=float)


class Model:
    """Fit and evaluate the one-parameter ring-current SCGF."""

    def __init__(self):
        self.diffusivity = 1.0

    def fit(self, records):
        """Fit diffusivity by bounded, sigma-weighted least squares.

        Each record supplies an ``input`` dictionary, a measured ``value``,
        and its known Gaussian standard deviation ``sigma``.  The weighted
        least-squares estimate is the Gaussian maximum-likelihood estimate;
        clipping it to the documented interval gives the constrained fit.
        """
        records = list(records)
        if not records:
            return self

        features = _counting_features([record["input"] for record in records])
        values = np.asarray([record["value"] for record in records], dtype=float)
        sigma = np.asarray([record["sigma"] for record in records], dtype=float)

        if (not np.all(np.isfinite(features)) or
                not np.all(np.isfinite(values)) or
                not np.all(np.isfinite(sigma)) or
                np.any(sigma <= 0.0)):
            raise ValueError("Calibration inputs, values, and positive sigmas must be finite.")

        scaled_features = features / sigma
        scaled_values = values / sigma
        denominator = float(np.dot(scaled_features, scaled_features))
        if denominator > 0.0 and np.isfinite(denominator):
            estimate = float(np.dot(scaled_features, scaled_values) / denominator)
            if not np.isfinite(estimate):
                raise ValueError("Calibration data do not produce a finite diffusivity estimate.")
            self.diffusivity = float(np.clip(estimate, *_DIFFUSIVITY_BOUNDS))
        else:
            # With zero features the records contain no information about D.
            # Keep the current valid estimate (normally the default of 1.0).
            self.diffusivity = float(np.clip(self.diffusivity, *_DIFFUSIVITY_BOUNDS))
        return self

    def predict(self, experiments):
        """Return one finite raw SCGF prediction per experiment."""
        return predict_at(experiments, self.diffusivity)
