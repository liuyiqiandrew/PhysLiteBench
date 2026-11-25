"""Scaled current cumulants for the half-filled weakly asymmetric ring.

The flat-density saddle gives a quadratic SCGF. At sufficiently large field,
the low-current saddle is an inhomogeneous, stationary density profile. We
maximize its macroscopic fluctuation theory functional in a Fourier basis.
The microscopic rate prefactor simply multiplies the resulting SCGF.
"""

from functools import lru_cache

import numpy as np
from scipy.optimize import minimize


_WAVE_NUM_MODES = 24
_WAVE_GRID_SIZE = 512
_MODES = 2 * np.arange(_WAVE_NUM_MODES) + 1
_POSITIONS = np.arange(_WAVE_GRID_SIZE) / _WAVE_GRID_SIZE
_BASIS = np.cos(2 * np.pi * _MODES[:, None] * _POSITIONS[None, :])
_WAVENUMBER_SQUARED = (2 * np.pi * _MODES) ** 2


@lru_cache(maxsize=1024)
def _unit_scgf(field, bias):
    """SCGF at D=1, including the stationary wave when it is favorable."""
    field = float(field)
    bias = float(bias)
    flat = bias * (bias + 2 * field) / 4

    # The first spatial Fourier mode becomes unstable at this boundary.
    if flat >= -np.pi**2:
        return flat

    shifted_bias_squared = (bias + field) ** 2
    field_squared = field**2

    def objective(coefficients):
        u = coefficients @ _BASIS
        cosine = np.cos(u)
        sine = np.sin(u)
        secant_squared = 1 / np.maximum(cosine**2, 1e-24)
        mean_secant_squared = np.mean(secant_squared)

        # The profile is rho=(1+sin(u))/2. Odd cosine harmonics enforce
        # half filling, periodicity and the wave's reflection symmetry.
        value = (
            shifted_bias_squared / (4 * mean_secant_squared)
            - field_squared / 4
            + field_squared * np.mean(sine**2) / 4
            - np.dot(_WAVENUMBER_SQUARED, coefficients**2) / 8
        )
        point_gradient = (
            -shifted_bias_squared * secant_squared * np.tan(u)
            / (2 * mean_secant_squared**2)
            + field_squared * np.sin(2 * u) / 4
        )
        gradient = _BASIS @ point_gradient / _WAVE_GRID_SIZE
        gradient -= _WAVENUMBER_SQUARED * coefficients / 4

        # u=arcsin(2 rho-1) lies inside [-pi/2, pi/2]. This penalty only
        # guards trial steps; the maximizing physical profiles are interior.
        excess = np.maximum(np.abs(u) - np.pi / 2, 0)
        if np.any(excess):
            value -= 1000 * np.mean(excess**2)
            gradient -= 2000 * (_BASIS @ (excess * np.sign(u))) / _WAVE_GRID_SIZE
        return -value, -gradient

    initial = np.zeros(_WAVE_NUM_MODES)
    initial[0] = 0.5
    result = minimize(
        objective, initial, jac=True, method="L-BFGS-B",
        options={"ftol": 1e-13, "gtol": 1e-9, "maxiter": 1000},
    )
    return max(flat, -float(result.fun))


def predict_at(experiments, diffusivity):
    """Predict the leading SCGF in inverse microscopic time units."""
    return float(diffusivity) * np.asarray(
        [_unit_scgf(e["field"], e["bias"]) for e in experiments], dtype=float
    )


class Model:
    def __init__(self):
        self.diffusivity = 1.0

    def fit(self, records):
        """Weighted Gaussian maximum-likelihood fit for the common rate D."""
        if not records:
            return self
        basis = np.asarray(
            [_unit_scgf(r["input"]["field"], r["input"]["bias"]) for r in records]
        )
        values = np.asarray([r["value"] for r in records], dtype=float)
        inverse_sigma = 1 / np.asarray([r["sigma"] for r in records], dtype=float)
        weighted_basis = basis * inverse_sigma
        denominator = float(weighted_basis @ weighted_basis)
        if denominator > 0:
            estimate = float(weighted_basis @ (values * inverse_sigma)) / denominator
            self.diffusivity = float(np.clip(estimate, 0.8, 1.2))
        return self

    def predict(self, experiments):
        return predict_at(experiments, self.diffusivity)
