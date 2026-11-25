"""Transition-rate model for the accelerated detector apparatus."""

import numpy as np


_TWO_PI = 2.0 * np.pi
_COUPLING_BOUNDS = (5.0e-5, 2.0e-4)


def _local_response(gap, acceleration):
    """Return the one-detector downward response, before multiplying by mu^2."""
    inertial = gap / _TWO_PI
    if acceleration == 0.0:
        return inertial

    # The acceleration-dependent factor is the Unruh thermal factor.  Using
    # expm1 keeps this well behaved when the acceleration is small.
    thermal_factor = -np.expm1(-_TWO_PI * gap / acceleration)
    return inertial / thermal_factor


def _cross_response(gap, acceleration, separation):
    """Return the cross-detector downward response.

    For the two parallel Rindler trajectories, the invariant separation is

        (Delta t)^2 - (Delta x)^2 - separation^2
        = 4/a^2 sinh^2(a Delta tau/2) - separation^2.

    Fourier transforming the corresponding Wightman function gives the
    expression below.  Its zero-acceleration limit is the usual transverse
    sinc response.
    """
    if acceleration == 0.0:
        # np.sinc(x/pi) is sin(x)/x.
        return _local_response(gap, acceleration) * np.sinc(
            gap * separation / np.pi
        )

    scaled_separation = 0.5 * acceleration * separation
    eta = np.arcsinh(scaled_separation)
    phase = (2.0 * gap / acceleration) * eta
    geometric_factor = np.sqrt(1.0 + scaled_separation * scaled_separation)
    thermal_factor = -np.expm1(-_TWO_PI * gap / acceleration)
    return (
        np.sin(phase)
        / (_TWO_PI * separation * geometric_factor * thermal_factor)
    )


def response(e):
    """Return the transition response with the coupling omitted."""
    gap = float(e["gap"])
    acceleration = float(e["acceleration"])

    local = _local_response(gap, acceleration)
    if e["preparation"] == "single":
        return local

    cross = _cross_response(gap, acceleration, float(e["separation"]))
    sign = 1.0 if e["preparation"] == "symmetric" else -1.0
    # The result is nonnegative analytically; max only protects against a
    # possible negative roundoff residue at an interference zero.
    return max(0.0, local + sign * cross)


def predict_at(experiments, coupling_squared):
    """Predict rates for experiments in their original input order."""
    return float(coupling_squared) * np.asarray(
        [response(experiment) for experiment in experiments], dtype=float
    )


class Model:
    def __init__(self):
        self.coupling_squared = None

    def fit(self, records):
        """Fit the common coupling using the supplied Gaussian calibration."""
        design = np.asarray([response(record["input"]) for record in records])
        values = np.asarray([record["value"] for record in records], dtype=float)
        sigma = np.asarray([record["sigma"] for record in records], dtype=float)

        # Rates are linear in q = mu^2.  This is the weighted least-squares
        # maximum-likelihood estimate for the stated independent Gaussian
        # measurements.
        weights = 1.0 / np.square(sigma)
        denominator = np.sum(weights * np.square(design))
        estimate = np.sum(weights * design * values) / denominator
        self.coupling_squared = float(np.clip(estimate, *_COUPLING_BOUNDS))
        return self

    def predict(self, experiments):
        return predict_at(experiments, self.coupling_squared)
