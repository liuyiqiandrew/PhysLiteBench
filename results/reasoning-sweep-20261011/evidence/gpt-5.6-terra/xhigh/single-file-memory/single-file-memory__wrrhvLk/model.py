"""Asymptotic covariance model for a tagged particle in a single file.

The initially equally spaced file is a *quenched* initial condition.  This is
important away from equal times: the covariance involves ``sqrt(t_a+t_b)``,
rather than the fractional-Brownian-motion form that applies after averaging
over random initial particle positions.
"""

import numpy as np


def covariance(diffusivity, density, time_a, time_b):
    """Return the long-time covariance coefficient for one experiment.

    For a deterministic file of density ``rho``, the tagged-particle result is

        sqrt(D) / (rho sqrt(pi))
        * [sqrt(t_a + t_b) - sqrt(|t_a - t_b|)].
    """
    diffusivity = float(diffusivity)
    density = float(density)
    time_a = float(time_a)
    time_b = float(time_b)

    if not (np.isfinite(diffusivity) and diffusivity >= 0.0):
        raise ValueError("diffusivity must be finite and non-negative")
    if not (np.isfinite(density) and density > 0.0):
        raise ValueError("density must be finite and positive")
    if not (np.isfinite(time_a) and np.isfinite(time_b) and time_a >= 0.0 and time_b >= 0.0):
        raise ValueError("times must be finite and non-negative")

    time_term = np.sqrt(time_a + time_b) - np.sqrt(abs(time_a - time_b))
    return float(np.sqrt(diffusivity / np.pi) * time_term / density)


def predict_at(experiments, diffusivity):
    """Evaluate a fitted diffusivity on experiments in their supplied order."""
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
        """Estimate the common diffusivity by Gaussian weighted least squares.

        The covariance is linear in ``sqrt(D)``, so the maximum-likelihood
        estimate has a closed form.
        """
        records = list(records)
        if not records:
            raise ValueError("at least one calibration record is required")

        design = np.empty(len(records), dtype=float)
        values = np.empty(len(records), dtype=float)
        sigmas = np.empty(len(records), dtype=float)

        for index, record in enumerate(records):
            experiment = record["input"]
            # covariance(1, ...) is the coefficient multiplying sqrt(D).
            design[index] = covariance(
                1.0,
                experiment["density"],
                experiment["time_a"],
                experiment["time_b"],
            )
            values[index] = float(record["value"])
            sigmas[index] = float(record["sigma"])

        if not np.isfinite(values).all():
            raise ValueError("calibration values must be finite")
        if not (np.isfinite(sigmas).all() and np.all(sigmas > 0.0)):
            raise ValueError("calibration sigmas must be finite and positive")

        weights = 1.0 / np.square(sigmas)
        denominator = np.dot(weights, np.square(design))
        if not np.isfinite(denominator) or denominator <= 0.0:
            raise ValueError("calibration experiments must contain time information")

        sqrt_diffusivity = np.dot(weights * design, values) / denominator
        # The supplied physics constrains the parameter; this is consequently
        # the constrained Gaussian maximum-likelihood estimate.
        sqrt_diffusivity = np.clip(sqrt_diffusivity, np.sqrt(0.8), np.sqrt(1.2))
        self.diffusivity = float(sqrt_diffusivity * sqrt_diffusivity)
        return self

    def predict(self, experiments):
        if self.diffusivity is None:
            raise RuntimeError("fit must be called before predict")
        return predict_at(experiments, self.diffusivity)
