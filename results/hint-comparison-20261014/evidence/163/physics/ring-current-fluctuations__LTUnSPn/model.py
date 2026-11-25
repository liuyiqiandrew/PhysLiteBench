"""Scaled current SCGF for the exclusion process on a ring."""

from functools import lru_cache

import numpy as np


_PI2_4 = 4.0 * np.pi * np.pi


def _homogeneous(field, bias):
    return bias * (bias + 2.0 * field) / 4.0


@lru_cache(maxsize=1024)
def _profile_value(field, bias):
    """Maximize the stationary ring variational problem.

    With rho=(1+sin(theta))/2, an odd cosine series for theta enforces the
    half-filling constraint.  The uniform profile is stable over most of the
    domain; when it is unstable, the Fourier maximization gives the periodic
    density profile selected by the long-time variational problem.
    """
    e = float(field)
    lam = float(bias)
    h = lam * (lam + 2.0 * e)
    if h >= -_PI2_4:
        return _homogeneous(e, lam)

    try:
        from scipy.optimize import minimize
    except Exception:  # pragma: no cover
        return _homogeneous(e, lam)

    modes = 2 * np.arange(9, dtype=float) + 1.0
    grid = np.arange(2048, dtype=float) / 2048.0
    basis = np.cos(2.0 * np.pi * grid[:, None] * modes[None, :])
    omega = 2.0 * np.pi * modes
    s = e + lam

    def objective(coefficients):
        theta = basis @ coefficients
        if np.max(np.abs(theta)) >= 0.5 * np.pi - 1.0e-7:
            return 1.0e12, np.zeros_like(coefficients)

        theta_x = -(basis * omega[None, :]) @ coefficients
        cosine = np.cos(theta)
        sine = np.sin(theta)
        inv_mobility = np.mean(1.0 / (cosine * cosine))
        mobility = np.mean(cosine * cosine)
        gradient_cost = np.mean(theta_x * theta_x)
        value = s * s / inv_mobility - e * e * mobility - gradient_cost

        d_inv = np.mean(
            (2.0 * sine / (cosine * cosine * cosine))[:, None] * basis,
            axis=0,
        )
        d_mobility = np.mean((-2.0 * cosine * sine)[:, None] * basis, axis=0)
        d_gradient = omega * omega * coefficients
        derivative = (
            -s * s * d_inv / (inv_mobility * inv_mobility)
            - e * e * d_mobility
            - d_gradient
        )
        return -value, -derivative

    # Keep the homogeneous branch as a candidate and use several amplitudes
    # so a weakly unstable point cannot settle at the uniform local maximum.
    best = s * s - e * e
    for amplitude in (0.03, 0.15, 0.35, 0.65, 1.0, 1.35):
        start = np.zeros(9, dtype=float)
        start[0] = amplitude
        result = minimize(
            objective,
            start,
            jac=True,
            method="L-BFGS-B",
            bounds=[(-1.5, 1.5)] * 9,
            options={
                "maxiter": 1200,
                "ftol": 1.0e-13,
                "gtol": 1.0e-9,
                "maxls": 40,
            },
        )
        candidate = -float(result.fun)
        if np.isfinite(candidate) and candidate > best:
            best = candidate

    return best / 4.0


def predict_at(experiments, diffusivity):
    """Return raw asymptotic SCGF values for a sequence of inputs."""
    if not experiments:
        return np.empty(0, dtype=float)
    values = np.asarray(
        [_profile_value(x["field"], x["bias"]) for x in experiments],
        dtype=float,
    )
    return np.asarray(float(diffusivity) * values, dtype=float)


class Model:
    def __init__(self):
        self.diffusivity = 1.0

    def fit(self, records):
        """Fit D by the stated independent-Gaussian weighted least squares."""
        design = []
        observations = []
        weights = []
        for record in records:
            inp = record["input"]
            sigma = float(record["sigma"])
            if not np.isfinite(sigma) or sigma <= 0.0:
                continue
            design.append(_profile_value(inp["field"], inp["bias"]))
            observations.append(float(record["value"]))
            weights.append(1.0 / (sigma * sigma))

        if design:
            x = np.asarray(design, dtype=float)
            y = np.asarray(observations, dtype=float)
            w = np.asarray(weights, dtype=float)
            denominator = float(np.sum(w * x * x))
            if denominator > 0.0 and np.isfinite(denominator):
                fitted = float(np.sum(w * x * y) / denominator)
                if np.isfinite(fitted):
                    self.diffusivity = float(np.clip(fitted, 0.8, 1.2))
        return self

    def predict(self, experiments):
        return predict_at(experiments, self.diffusivity)
