"""Prediction model for the accelerated-detector apparatus."""

import numpy as np


_TWO_PI = 2.0 * np.pi


def _local_response(gap, acceleration):
    """Downward transition rate with the coupling squared removed."""
    inertial = gap / _TWO_PI
    if acceleration == 0.0:
        return inertial
    return inertial / (-np.expm1(-_TWO_PI * gap / acceleration))


def _cross_response_factor(gap, acceleration, separation):
    """Cross-detector spectrum divided by the local spectrum.

    For b=a*d/2 and q=asinh(b), Fourier transformation of the cross
    Wightman function gives the factor below.  Its continuous inertial limit
    is sinc(gap*d).
    """
    if acceleration == 0.0:
        return np.sinc(gap * separation / np.pi)

    b = 0.5 * acceleration * separation
    q = np.arcsinh(b)
    if separation == 0.0:
        return 1.0

    phase = (2.0 * gap / acceleration) * q
    denominator = gap * separation * np.sqrt(1.0 + b * b)
    return np.sin(phase) / denominator


def response(e):
    """Return the unit-coupling rate for one experiment."""
    gap = float(e["gap"])
    acceleration = float(e["acceleration"])
    local = _local_response(gap, acceleration)
    preparation = e["preparation"]
    if preparation == "single":
        return float(local)

    separation = float(e["separation"])
    cross = _cross_response_factor(gap, acceleration, separation)
    if preparation == "symmetric":
        return float(local * (1.0 + cross))
    if preparation == "antisymmetric":
        return float(local * (1.0 - cross))
    raise ValueError(f"unknown preparation: {preparation!r}")


def predict_at(experiments, coupling_squared):
    """Predict rates in the same order as ``experiments``."""
    return np.asarray(
        [float(coupling_squared) * response(experiment) for experiment in experiments],
        dtype=float,
    )


class Model:
    def __init__(self):
        self.coupling_squared = None

    def fit(self, records):
        """Fit the common coupling squared by weighted least squares."""
        design = np.asarray([response(record["input"]) for record in records], dtype=float)
        values = np.asarray([record["value"] for record in records], dtype=float)
        sigmas = np.asarray([record["sigma"] for record in records], dtype=float)

        weights = 1.0 / np.square(sigmas)
        denominator = np.sum(weights * design * design)
        if denominator == 0.0 or not np.isfinite(denominator):
            raise ValueError("calibration records do not determine the coupling")

        estimate = np.sum(weights * design * values) / denominator
        if not np.isfinite(estimate):
            raise ValueError("calibration produced a non-finite coupling")

        self.coupling_squared = float(estimate)
        return self

    def predict(self, experiments):
        if self.coupling_squared is None:
            raise ValueError("fit the model before prediction")
        return predict_at(experiments, self.coupling_squared)
