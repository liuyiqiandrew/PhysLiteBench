"""Tagged-particle covariance model.

Setup
-----
Identical Brownian point particles (diffusion constant ``D``) diffuse on the
line with perfectly reflecting collisions, starting from the deterministic,
equally spaced configuration ``x_j(0) = j / density``.  The particle at the
origin is tagged and followed without any force being exerted on it.  The
reported quantity is the joint long-time limit

    lim_{lambda -> infinity} Cov[X(lambda*time_a), X(lambda*time_b)] / sqrt(lambda),

for fixed positive ``time_a`` and ``time_b``.

Derivation
----------
The collision dynamics of identical point particles is order-preserving, so the
tagged particle is exactly the particle of rank 0 of the sorted free-particle
positions, i.e. the ``(n+1)``-th order statistic of the free walkers.

For an order statistic with uniform local density ``density``, the fluctuations
of the median path are governed by the net number of *independent* Brownian
paths that have crossed the reference level.  In the joint long-time scaling
above this reduces to

    X(lambda*t) / sqrt(lambda)  ~  -B(t) / density  (in law, modulo the marked
    particle's own free displacement, whose covariance contributes at lower
    order),

where ``B`` is a single Brownian motion (with the order-statistic diffusion
constant).  Equivalently, the two-time covariance of ``X`` is controlled by the
crossing indicators ``1{Z_{t_a} < 0}`` and ``1{Z_{t_b} < 0}`` of one *shared*
path, so the two indicators remain correlated even as ``lambda -> infinity``.

Carrying out the long-time limit of the order-statistic two-time covariance
gives the closed form used below,

    Cov/sqrt(lambda) = sqrt(D/pi)/density
                      * ( sqrt(time_a + time_b) - sqrt(|time_a - time_b|) ),

which reproduces the equal-time (``time_a == time_b``) results
``sqrt(2 D time / pi) / density`` that the calibration records supply, and
extends them consistently to unequal times.

All of the calibration records have ``time_a == time_b``; the equal-time data
therefore fixes only the amplitude, and the documented joint two-time limit
fixes the remaining time dependence as implemented here.
"""

import numpy as np


def covariance(diffusivity, density, time_a, time_b):
    """Long-time limiting covariance coefficient (in ``L0**2``).

    Parameters
    ----------
    diffusivity : float
        Common diffusion constant ``D`` in ``L0**2 / T0``.
    density : float
        Initial number of particles per unit length in ``1 / L0``.
    time_a, time_b : float
        Fixed positive time multipliers in ``T0``.

    Returns
    -------
    float
        ``lim_{lambda->inf} Cov[X(lambda*time_a), X(lambda*time_b)] / sqrt(lambda)``.
    """
    diffusivity = float(diffusivity)
    density = float(density)
    time_a = float(time_a)
    time_b = float(time_b)

    amplitude = np.sqrt(diffusivity / np.pi) / density
    time_factor = np.sqrt(time_a + time_b) - np.sqrt(abs(time_a - time_b))
    return amplitude * time_factor


def predict_at(experiments, diffusivity):
    """Vectorised predictions for an iterable of experiment dictionaries."""
    return np.asarray(
        [
            covariance(
                float(diffusivity),
                float(e["density"]),
                float(e["time_a"]),
                float(e["time_b"]),
            )
            for e in experiments
        ],
        dtype=float,
    )


class Model:
    """Calibrated model of the tagged-particle covariance coefficient."""

    def __init__(self):
        self.diffusivity = None

    def fit(self, records):
        """Infer the common diffusivity from calibration records.

        The predicted coefficient is ``sqrt(D) * g`` with the known basis
        function ``g`` (for the calibration records), so the model is linear
        in ``sqrt(D)``.  A weighted least-squares solve gives the Gaussian
        maximum-likelihood estimate of ``D``.
        """
        records = list(records)
        if not records:
            raise ValueError("records must contain at least one calibration entry")

        density = np.array([float(r["input"]["density"]) for r in records])
        time_a = np.array([float(r["input"]["time_a"]) for r in records])
        time_b = np.array([float(r["input"]["time_b"]) for r in records])
        value = np.array([float(r["value"]) for r in records])
        sigma = np.array([float(r["sigma"]) for r in records])

        # value = sqrt(D) * g, with g the known per-record basis.
        g = (
            np.sqrt(time_a + time_b) - np.sqrt(np.abs(time_a - time_b))
        ) / (density * np.sqrt(np.pi))

        weights = 1.0 / sigma**2
        denom = float(np.sum(weights * g * g))
        if denom <= 0.0 or not np.isfinite(denom):
            raise ValueError("calibration records do not constrain the diffusivity")
        root_diffusivity = float(np.sum(weights * g * value) / denom)

        # The diffusion constant is non-negative; guard against a negative
        # weighted estimate caused by extreme noise.
        root_diffusivity = max(root_diffusivity, 0.0)
        self.diffusivity = root_diffusivity**2
        return self

    def predict(self, experiments):
        """Predict the covariance coefficients for the given experiments."""
        if self.diffusivity is None:
            raise ValueError("call fit before predict")
        return predict_at(experiments, self.diffusivity)
