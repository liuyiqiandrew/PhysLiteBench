"""Scaled current generating function for the half-filled weakly asymmetric ring."""

from functools import lru_cache

import numpy as np
from scipy.optimize import minimize


_CRITICAL = -4.0 * np.pi**2
_MODES = 10
_GRID = 256


@lru_cache(maxsize=1)
def _basis():
    """Odd Fourier modes impose half filling and particle-hole symmetry."""
    positions = np.arange(_GRID) / _GRID
    wave_numbers = 2 * np.arange(_MODES) + 1
    angles = 2 * np.pi * positions[:, None] * wave_numbers
    return np.cos(angles), -2 * np.pi * wave_numbers * np.sin(angles)


@lru_cache(maxsize=1024)
def _unit_scgf(field, shifted_bias_squared):
    """Optimize the stationary macroscopic fluctuation functional at D=1.

    For a density profile rho(x), put a=rho(1-rho). Eliminating the
    space-independent current gives

        (E+lambda)^2 / integral(1/a)
        - integral(rho'^2/a)/4 - E^2 integral(a).

    The optimal profile at half filling is particle-hole symmetric. Writing
    rho=(1+sin(theta))/2 makes rho'^2/a=theta'^2. A tanh parameterization
    keeps the numerical density strictly between zero and one.
    """
    e2 = field * field
    b = shifted_bias_squared
    flat = (b - e2) / 4.0
    if b - e2 >= _CRITICAL:
        return flat

    basis, basis_prime = _basis()

    def objective(coefficients):
        z = basis @ coefficients
        z_prime = basis_prime @ coefficients
        tanh_z = np.tanh(z)
        theta = (np.pi / 2) * tanh_z
        theta_factor = (np.pi / 2) * (1 - tanh_z**2)
        theta_prime = theta_factor * z_prime
        cosine = np.cos(theta)
        secant_squared = 1 / cosine**2
        inverse_mobility = np.mean(secant_squared)

        cost = (np.mean(theta_prime**2 + e2 * cosine**2)
                - b / inverse_mobility) / 4

        # Gradient of the cost with respect to the Fourier coefficients.
        local = theta_factor * (
            -e2 * np.sin(2 * theta) / 4
            + b * secant_squared * np.tan(theta)
            / (2 * inverse_mobility**2)
        )
        gradient = np.mean(
            (theta_prime * theta_factor)[:, None] * basis_prime / 2
            + (local - theta_prime * theta_factor * tanh_z * z_prime)[:, None]
            * basis,
            axis=0,
        )
        return cost, gradient

    start = np.zeros(_MODES)
    start[0] = 0.7
    result = minimize(
        objective,
        start,
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": 500, "ftol": 1e-13, "gtol": 1e-9},
    )
    return max(flat, -float(result.fun))


def predict_at(experiments, diffusivity):
    """Predict raw SCGF values for an iterable of field/bias dictionaries."""
    pairs = [(float(item["field"]), float(item["bias"]))
             for item in experiments]
    values = np.empty(len(pairs), dtype=float)
    for index, (field, bias) in enumerate(pairs):
        values[index] = diffusivity * _unit_scgf(field, (field + bias)**2)
    return values


class Model:
    def __init__(self):
        self.diffusivity = 1.0

    def fit(self, records):
        records = list(records)
        if not records:
            return self
        unit_values = predict_at((item["input"] for item in records), 1.0)
        observed = np.asarray([item["value"] for item in records], dtype=float)
        sigma = np.asarray([item["sigma"] for item in records], dtype=float)
        scaled = unit_values / sigma
        information = np.dot(scaled, scaled)
        if information > 0:
            estimate = np.dot(scaled, observed / sigma) / information
            self.diffusivity = float(np.clip(estimate, 0.8, 1.2))
        return self

    def predict(self, experiments):
        return predict_at(experiments, self.diffusivity)
