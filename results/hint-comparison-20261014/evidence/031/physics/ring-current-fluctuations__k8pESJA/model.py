"""Macroscopic current SCGF for the weakly driven exclusion ring.

The spatially uniform answer is correct only while that saddle is stable.
Beyond the current-conditioning transition the optimal state is a density wave
(its arbitrary position is why the unconditioned one-point density is 1/2).
"""

from __future__ import annotations

import numpy as np


_FOUR_PI_SQUARED = 4.0 * np.pi * np.pi


def _uniform_scgf(field: float, bias: float) -> float:
    """SCGF with diffusivity set to one, for the flat saddle."""
    return 0.25 * bias * (bias + 2.0 * field)


def _modulated_scgf(field: float, bias: float) -> float:
    """Numerically maximize the stationary MFT functional at half filling.

    With ``u=rho-1/2`` and ``s=2 rho (1-rho)``, eliminating the constant
    current gives (in units of diffusivity)

        (E+lambda)^2 / (2 <1/s>) - <u'^2/s>/2 - E^2 <s>/2.

    The half-filled density-wave branch obeys u(x+1/2)=-u(x).  Parameterizing
    it as ``u=tanh(v)/2`` enforces density bounds without constraining its
    amplitude or its translation-degenerate saddle.
    """
    from scipy.optimize import minimize

    e = float(field)
    k = e + float(bias)
    modes = 12
    grid_size = 256
    x = np.arange(grid_size, dtype=float) / grid_size
    n = np.arange(1, 2 * modes, 2, dtype=float)
    cosines = np.cos(2.0 * np.pi * x[:, None] * n)
    derivatives = -2.0 * np.pi * n * np.sin(2.0 * np.pi * x[:, None] * n)

    def value_and_gradient(coefficients: np.ndarray):
        v = cosines @ coefficients
        v_prime = derivatives @ coefficients
        tanh_v = np.tanh(v)
        sech2_v = 1.0 - tanh_v * tanh_v
        u = 0.5 * tanh_v
        u_prime = 0.5 * sech2_v * v_prime
        mobility = 0.5 - 2.0 * u * u

        inverse_mobility = np.mean(1.0 / mobility)
        gradient_cost = np.mean(u_prime * u_prime / mobility)
        mean_mobility = np.mean(mobility)
        value = (k * k / (2.0 * inverse_mobility)
                 - 0.5 * gradient_cost - 0.5 * e * e * mean_mobility)

        du = 0.5 * sech2_v[:, None] * cosines
        du_prime = (0.5 * sech2_v[:, None] * derivatives
                    - sech2_v[:, None] * tanh_v[:, None]
                    * v_prime[:, None] * cosines)
        dmobility = -4.0 * u[:, None] * du
        d_inverse = np.mean(-dmobility / mobility[:, None] ** 2, axis=0)
        d_gradient = np.mean(
            2.0 * u_prime[:, None] * du_prime / mobility[:, None]
            - u_prime[:, None] ** 2 * dmobility / mobility[:, None] ** 2,
            axis=0,
        )
        d_mean_mobility = np.mean(dmobility, axis=0)
        gradient = (-k * k * d_inverse / (2.0 * inverse_mobility ** 2)
                    - 0.5 * d_gradient - 0.5 * e * e * d_mean_mobility)
        return float(value), gradient

    # Zero is the homogeneous stationary point.  Nonzero seeds are needed to
    # find the better density-wave maximum after it loses stability.
    best = _uniform_scgf(e, bias)
    for amplitude in (0.05, 0.35, 1.0, 2.0):
        start = np.zeros(modes)
        start[0] = amplitude
        result = minimize(
            lambda a: -value_and_gradient(a)[0], start,
            jac=lambda a: -value_and_gradient(a)[1], method="L-BFGS-B",
            bounds=[(-6.0, 6.0)] * modes,
            options={"ftol": 1e-13, "gtol": 1e-10, "maxiter": 1200,
                     "maxls": 80},
        )
        candidate = -float(result.fun)
        if np.isfinite(candidate):
            best = max(best, candidate)
    return best


def _unit_scgf(field: float, bias: float) -> float:
    """Return psi/D, including the density-wave phase when it is favorable."""
    e = float(field)
    lam = float(bias)
    # The first spatial Fourier mode has curvature proportional to
    # E^2-(E+lambda)^2-4*pi^2 around rho=1/2.
    if e * e - (e + lam) * (e + lam) <= _FOUR_PI_SQUARED:
        return _uniform_scgf(e, lam)
    return _modulated_scgf(e, lam)


def predict_at(experiments, diffusivity):
    """Predict raw SCGF values for an iterable of documented input records."""
    experiments = list(experiments)
    if not experiments:
        return np.empty(0, dtype=float)

    cache: dict[tuple[float, float], float] = {}
    output = np.empty(len(experiments), dtype=float)
    for index, experiment in enumerate(experiments):
        key = (float(experiment["field"]), float(experiment["bias"]))
        if key not in cache:
            cache[key] = _unit_scgf(*key)
        output[index] = float(diffusivity) * cache[key]
    return output


class Model:
    def __init__(self):
        self.diffusivity = 1.0

    def fit(self, records):
        """Fit the single common rate prefactor by weighted least squares."""
        records = list(records)
        if not records:
            self.diffusivity = 1.0
            return self

        unit_values = np.asarray(
            [_unit_scgf(record["input"]["field"], record["input"]["bias"])
             for record in records], dtype=float)
        observations = np.asarray([record["value"] for record in records], dtype=float)
        sigmas = np.asarray([record["sigma"] for record in records], dtype=float)
        valid = (np.isfinite(unit_values) & np.isfinite(observations)
                 & np.isfinite(sigmas) & (sigmas > 0.0))
        weights = np.zeros_like(sigmas)
        weights[valid] = 1.0 / sigmas[valid] ** 2
        denominator = float(np.sum(weights * unit_values * unit_values))
        if denominator > 0.0 and np.isfinite(denominator):
            estimate = float(np.sum(weights * unit_values * observations) / denominator)
            if np.isfinite(estimate):
                self.diffusivity = float(np.clip(estimate, 0.8, 1.2))
                return self
        self.diffusivity = 1.0
        return self

    def predict(self, experiments):
        return predict_at(experiments, self.diffusivity)
