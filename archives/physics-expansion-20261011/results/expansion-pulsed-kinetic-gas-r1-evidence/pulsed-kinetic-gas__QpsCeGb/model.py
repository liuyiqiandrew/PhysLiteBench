"""Analytical model for the pulsed gas density experiment."""

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import jv


def _displacement_variance(t, friction, thermal_speed):
    """Variance of the free OU displacement over an interval of length t."""
    g = float(friction)
    return 2.0 * thermal_speed**2 * (g * t + np.expm1(-g * t)) / g**2


def response(e, gamma, cutoff=64):
    """Return one signed Fourier component for an experiment."""
    tau = float(e["delay"])
    after = float(e["after"])
    speed = float(e["thermal_speed"])
    g = float(gamma)

    def displacement_gain(t):
        return -np.expm1(-g * t) / g

    gain_delay = displacement_gain(tau)
    gain_after = displacement_gain(after)
    gain_total = displacement_gain(tau + after)

    var_delay = _displacement_variance(tau, g, speed)
    var_after = _displacement_variance(after, g, speed)
    # The free displacement before the second kick is correlated with the
    # displacement after it because the OU velocity retains memory.
    covariance = speed**2 * gain_delay * gain_after

    k1 = int(e["k1"])
    k2 = int(e["k2"])
    mode = int(e["mode"])

    # Expand each sinusoidal kick in Bessel functions and retain the terms
    # surviving the uniform average over the initial position.
    n = np.arange(-int(cutoff), int(cutoff) + 1, dtype=int)
    keep = (mode + n * k2) % k1 == 0
    n = n[keep]
    ell = -(mode + n * k2) // k1
    q = mode + n * k2

    # q multiplies the pre-delay displacement and mode the post-delay one.
    variance = (
        q * q * var_delay
        + mode * mode * var_after
        + 2.0 * q * mode * covariance
    )
    first_argument = float(e["a1"]) * (
        mode * gain_total + n * k2 * gain_delay
    )
    second_argument = mode * float(e["a2"]) * gain_after
    terms = (
        jv(n, second_argument)
        * jv(ell, first_argument)
        * np.exp(
            -0.5 * variance
            + 1j * (n * float(e["phase2"]) + ell * float(e["phase1"]))
        )
    )
    value = np.sum(terms)
    return float(value.real if e["component"] == "cos" else value.imag)


def predict_at(experiments, friction):
    return np.asarray(
        [response(experiment, float(friction)) for experiment in experiments],
        dtype=float,
    )


class Model:
    def __init__(self):
        self.friction = None

    def fit(self, records):
        """Estimate the common friction by weighted least squares."""
        records = list(records)
        if not records:
            raise ValueError("at least one calibration record is required")

        experiments = [record["input"] for record in records]
        observed = np.asarray([record["value"] for record in records], dtype=float)
        sigma = np.asarray([record["sigma"] for record in records], dtype=float)
        if np.any(~np.isfinite(observed)) or np.any(~np.isfinite(sigma)):
            raise ValueError("calibration values and sigmas must be finite")
        if np.any(sigma <= 0.0):
            raise ValueError("calibration sigmas must be positive")

        def objective(friction):
            residual = (predict_at(experiments, friction) - observed) / sigma
            return float(np.dot(residual, residual))

        result = minimize_scalar(
            objective,
            bounds=(0.08, 0.4),
            method="bounded",
            options={"xatol": 1e-12},
        )
        candidates = [float(result.x), 0.08, 0.4]
        self.friction = float(min(candidates, key=objective))
        return self

    def predict(self, experiments):
        if self.friction is None:
            raise RuntimeError("fit the model before predicting")
        return predict_at(experiments, self.friction)
