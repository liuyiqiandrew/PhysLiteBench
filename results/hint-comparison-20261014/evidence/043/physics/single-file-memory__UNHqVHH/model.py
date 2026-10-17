"""Asymptotic covariance model for a single-file Brownian system.

The preparation in this problem is a fixed lattice, rather than an
equilibrium (Poisson) preparation.  That distinction matters for covariance
at two different times.
"""

import numpy as np


def covariance(diffusivity, density, time_a, time_b):
    """Return the long-time covariance coefficient for one experiment.

    At leading order the marked-particle displacement is the negative of the
    current through its initial position, divided by ``density``.  For a
    Brownian path started at a fixed position, the integral of its two-time
    crossing-indicator covariance is

        sqrt(D / pi) * (sqrt(a + b) - sqrt(|a - b|)).

    Only crossings of the *same* independent path are correlated here.  In
    particular this is not the stationary-increment expression appropriate
    to a different initial ensemble.
    """
    diffusivity = float(diffusivity)
    density = float(density)
    time_a = float(time_a)
    time_b = float(time_b)
    return (
        np.sqrt(diffusivity / np.pi)
        * (np.sqrt(time_a + time_b) - np.sqrt(abs(time_a - time_b)))
        / density
    )


def predict_at(experiments, diffusivity):
    """Evaluate the documented experiment dictionaries in their input order."""
    return np.asarray(
        [
            covariance(
                diffusivity,
                experiment["density"],
                experiment["time_a"],
                experiment["time_b"],
            )
            for experiment in experiments
        ],
        dtype=float,
    )


class Model:
    def __init__(self):
        self.diffusivity = None

    def fit(self, records):
        """Fit the common diffusivity by Gaussian weighted least squares.

        The prediction is linear in ``sqrt(D)``.  Estimating that parameter
        directly gives the (constrained) maximum-likelihood estimate under
        the supplied independent Gaussian measurement errors.
        """
        records = list(records)
        if not records:
            raise ValueError("at least one calibration record is required")

        factors = np.asarray(
            [
                covariance(1.0, r["input"]["density"],
                           r["input"]["time_a"], r["input"]["time_b"])
                for r in records
            ],
            dtype=float,
        )
        values = np.asarray([r["value"] for r in records], dtype=float)
        sigmas = np.asarray([r["sigma"] for r in records], dtype=float)
        if (not np.isfinite(factors).all() or not np.isfinite(values).all()
                or not np.isfinite(sigmas).all() or np.any(sigmas <= 0)):
            raise ValueError("calibration records must contain finite values and positive sigma")

        weights = 1.0 / sigmas**2
        denominator = np.sum(weights * factors**2)
        if not np.isfinite(denominator) or denominator <= 0:
            raise ValueError("calibration records do not identify diffusivity")

        root_diffusivity = np.sum(weights * factors * values) / denominator
        # D is known to be in the stated physical range.  The MLE is therefore
        # clipped in sqrt(D), the parameter in which the model is linear.
        root_diffusivity = np.clip(root_diffusivity, np.sqrt(0.8), np.sqrt(1.2))
        self.diffusivity = float(root_diffusivity**2)
        return self

    def predict(self, experiments):
        if self.diffusivity is None:
            raise RuntimeError("fit must be called before predict")
        return predict_at(experiments, self.diffusivity)
