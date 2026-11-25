"""Macroscopic-fluctuation model for the weakly driven exclusion ring.

The diffusivity multiplies the generator and therefore also the SCGF.  The
usual homogeneous answer is not always the winning saddle: a modulated
half-filled profile takes over for sufficiently current-opposing biases.
"""

from __future__ import annotations

import numpy as np

try:  # scipy is available in the evaluation environment; retain a fallback.
    from scipy.optimize import minimize
except ImportError:  # pragma: no cover
    minimize = None


_TWO_PI = 2.0 * np.pi
_INSTABILITY = _TWO_PI ** 2


def _modulated_unit_scgf(field: float, bias: float) -> float:
    """Optimize the spatially modulated, half-filled MFT saddle.

    With m=2*rho-1, the wave speed vanishes at half filling.  Optimizing the
    current gives K=(E+bias)/int[1/(1-m^2)], and the remaining functional is

      bias*K/2 - 1/4 int ((m')^2 + (K-E+E*m^2)^2)/(1-m^2).

    An odd-harmonic tanh series enforces the physical density bounds and the
    zero-density constraint while fixing the arbitrary translation of a wave.
    """
    harmonics = np.arange(1, 18, 2, dtype=float)
    grid = np.arange(512, dtype=float) / 512.0
    angles = _TWO_PI * harmonics[:, None] * grid[None, :]
    cosine = np.cos(angles)
    derivative = -_TWO_PI * harmonics[:, None] * np.sin(angles)
    homogeneous = bias * (bias + 2.0 * field) / 4.0

    def negative_value(coefficients: np.ndarray) -> float:
        u = coefficients @ cosine
        u_prime = coefficients @ derivative
        m = np.tanh(u)
        one_minus_m2 = 1.0 - m * m
        m_prime = one_minus_m2 * u_prime
        inverse_mobility_mean = np.mean(1.0 / one_minus_m2)
        k = (field + bias) / inverse_mobility_mean
        c = k - field
        action = 0.25 * np.mean(
            (m_prime * m_prime + (c + field * m * m) ** 2)
            / one_minus_m2
        )
        return -(0.5 * bias * k - action)

    if minimize is None:
        return homogeneous

    best = homogeneous
    # The uniform point remains stationary beyond the transition, hence the
    # finite-amplitude initial states.
    for amplitude in (0.08, 0.25, 0.7, 1.5, 3.0):
        initial = np.zeros(len(harmonics), dtype=float)
        initial[0] = amplitude
        result = minimize(
            negative_value, initial, method="BFGS",
            options={"gtol": 2e-9, "maxiter": 500},
        )
        candidate = -float(result.fun)
        if np.isfinite(candidate) and candidate > best:
            best = candidate
    return best


def _unit_scgf(field: float, bias: float) -> float:
    """SCGF at unit diffusivity, including the modulated phase."""
    homogeneous = bias * (bias + 2.0 * field) / 4.0
    # The first ring Fourier mode is 2*pi.  Below this stability boundary the
    # homogeneous solution is the global saddle.
    if bias * (bias + 2.0 * field) >= -_INSTABILITY:
        return homogeneous
    return _modulated_unit_scgf(field, bias)


def predict_at(experiments, diffusivity):
    """Predict raw (not standardized) SCGFs for an iterable of inputs."""
    experiments = list(experiments)
    if not experiments:
        return np.empty(0, dtype=float)

    values = np.empty(len(experiments), dtype=float)
    cache: dict[tuple[float, float], float] = {}
    for index, experiment in enumerate(experiments):
        field = float(experiment["field"])
        bias = float(experiment["bias"])
        key = (field, bias)
        unit_value = cache.get(key)
        if unit_value is None:
            unit_value = _unit_scgf(field, bias)
            cache[key] = unit_value
        values[index] = float(diffusivity) * unit_value
    return values


class Model:
    def __init__(self):
        self.diffusivity = 1.0

    def fit(self, records):
        """Fit the common clock-rate multiplier by weighted least squares."""
        records = list(records)
        if not records:
            self.diffusivity = 1.0
            return self

        unit = np.asarray(
            [_unit_scgf(float(r["input"]["field"]), float(r["input"]["bias"]))
             for r in records], dtype=float,
        )
        observed = np.asarray([r["value"] for r in records], dtype=float)
        sigma = np.asarray([r["sigma"] for r in records], dtype=float)
        weights = 1.0 / (sigma * sigma)
        denominator = float(np.sum(weights * unit * unit))
        if np.isfinite(denominator) and denominator > 0.0:
            estimate = float(np.sum(weights * unit * observed) / denominator)
            if np.isfinite(estimate):
                self.diffusivity = float(np.clip(estimate, 0.8, 1.2))
        return self

    def predict(self, experiments):
        return predict_at(experiments, self.diffusivity)
