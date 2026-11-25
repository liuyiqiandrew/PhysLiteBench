"""Transition rates for the detector apparatus described in ``README.md``."""

import numpy as np


_COUPLING_MIN = 0.00005
_COUPLING_MAX = 0.0002


def _local_response(gap, acceleration):
    """Fourier transform of the pullback of the local Wightman function."""
    inertial = gap / (2.0 * np.pi)
    if acceleration == 0:
        return inertial

    # expm1 is accurate when the acceleration is small and avoids loss of
    # precision in the thermal factor.
    return inertial / (-np.expm1(-2.0 * np.pi * gap / acceleration))


def _cross_ratio(gap, acceleration, separation):
    """Cross response divided by the local response.

    For equal uniformly accelerated trajectories with transverse separation
    d, Fourier transformation of the cross Wightman function gives this
    expression.  Its zero-acceleration limit is sinc(gap*d).
    """
    if acceleration == 0:
        return np.sinc(gap * separation / np.pi)

    q = acceleration * separation / 2.0
    alpha = np.arcsinh(q)
    return np.sin((2.0 * gap / acceleration) * alpha) / (
        gap * separation * np.sqrt(1.0 + q * q)
    )


def response(e):
    """Return the leading-order rate with the coupling factor removed."""
    gap = float(e["gap"])
    acceleration = float(e["acceleration"])
    separation = float(e["separation"])

    local = _local_response(gap, acceleration)
    if e["preparation"] == "single":
        return local

    cross = _cross_ratio(gap, acceleration, separation)
    sign = 1.0 if e["preparation"] == "symmetric" else -1.0
    return local * (1.0 + sign * cross)


def predict_at(experiments, coupling_squared):
    """Predict rates in input order as a one-dimensional NumPy array."""
    values = np.asarray(
        [coupling_squared * response(experiment) for experiment in experiments],
        dtype=float,
    )
    return values.reshape(-1)


class Model:
    def __init__(self):
        self.coupling_squared = None

    def fit(self, records):
        """Estimate the shared coupling with inverse-variance weighting."""
        design = np.asarray([response(record["input"]) for record in records])
        values = np.asarray([record["value"] for record in records], dtype=float)
        sigmas = np.asarray([record["sigma"] for record in records], dtype=float)

        weights = 1.0 / (sigmas * sigmas)
        estimate = np.sum(weights * design * values) / np.sum(
            weights * design * design
        )
        self.coupling_squared = float(
            np.clip(estimate, _COUPLING_MIN, _COUPLING_MAX)
        )
        return self

    def predict(self, experiments):
        return predict_at(experiments, self.coupling_squared)
