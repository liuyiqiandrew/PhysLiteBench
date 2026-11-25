"""Macroscopic current SCGF for the half-filled weakly asymmetric SEP.

The field and counting field both scale as ``1 / L`` microscopically, so the
answer is proportional to the microscopic diffusivity.  In the uniform phase
its dimensionless part is ``bias * (bias + 2 * field) / 4``.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np


_TWO_PI_SQUARED = (2.0 * np.pi) ** 2


def _homogeneous_scgf(field: np.ndarray, bias: np.ndarray) -> np.ndarray:
    return bias * (bias + 2.0 * field) / 4.0


def _is_modulated(field: float, bias: float) -> bool:
    """Test stability of the uniform saddle against its first Fourier mode."""
    return field * field - (field + bias) ** 2 > _TWO_PI_SQUARED + 1.0e-10


@lru_cache(maxsize=512)
def _modulated_scgf(field: float, bias: float) -> float:
    """Dimensionless SCGF in the high-field, density-wave phase.

    At half filling particle-hole symmetry makes the optimal travelling wave
    stationary.  For a density profile r(x), its rate functional is

      integral ((q - 2 E r(1-r))**2 + r'**2) / (4 r(1-r)) dx.

    Maximising ``bias*q - I`` is done on a periodic grid.  This branch is only
    entered above E=2*pi, outside the supplied calibration range.
    """
    try:
        from scipy.optimize import minimize
    except ImportError:
        # Preserve a finite API in a minimal NumPy-only runtime.
        return float(_homogeneous_scgf(np.asarray(field), np.asarray(bias)))

    n = 128
    dx = 1.0 / n
    index = np.arange(n, dtype=float)
    uniform_q = 0.5 * (field + bias)
    homogeneous = float(_homogeneous_scgf(np.asarray(field), np.asarray(bias)))

    def objective_and_jacobian(x: np.ndarray):
        rho = x[:-2]
        q = x[-2]
        velocity = x[-1]
        mobility = 2.0 * rho * (1.0 - rho)
        mobility_prime = 2.0 - 4.0 * rho
        rho_prime = (np.roll(rho, -1) - rho) / dx
        residual = q + velocity * (rho - 0.5) - field * mobility
        rate = dx * np.sum((residual * residual + rho_prime * rho_prime) / (2.0 * mobility))

        # Analytic derivative of I - bias*q, including both neighbours of a
        # forward-difference link.
        direct = (
            residual * (velocity - field * mobility_prime) / mobility
            - (residual * residual + rho_prime * rho_prime)
            * mobility_prime / (2.0 * mobility * mobility)
        )
        link = rho_prime / mobility
        grad_rho = dx * direct - link + np.roll(link, 1)
        grad_q = dx * np.sum(residual / mobility) - bias
        grad_velocity = dx * np.sum(residual * (rho - 0.5) / mobility)
        return rate - bias * q, np.concatenate((grad_rho, [grad_q, grad_velocity]))

    constraint = {
        "type": "eq",
        "fun": lambda x: float(np.mean(x[:-2]) - 0.5),
        "jac": lambda x: np.concatenate((np.full(n, 1.0 / n), [0.0, 0.0])),
    }
    bounds = [(1.0e-5, 1.0 - 1.0e-5)] * n + [(-40.0, 40.0), (-100.0, 100.0)]
    candidates = [homogeneous]
    # Multiple amplitudes prevent a particular optimiser run from lingering at
    # the unstable uniform stationary point.
    for amplitude in (0.015, 0.075, 0.22, 0.42):
        rho0 = 0.5 + amplitude * np.cos(2.0 * np.pi * index / n)
        start = np.concatenate((rho0, [uniform_q, 0.0]))
        result = minimize(
            objective_and_jacobian,
            start,
            jac=True,
            method="SLSQP",
            bounds=bounds,
            constraints=constraint,
            options={"maxiter": 600, "ftol": 2.0e-10},
        )
        if np.isfinite(result.fun):
            candidates.append(float(-result.fun))
    return float(max(candidates))


def predict_at(experiments, diffusivity):
    """Return raw SCGF values for an iterable of documented input mappings."""
    experiments = list(experiments)
    if not experiments:
        return np.empty(0, dtype=float)
    field = np.asarray([item["field"] for item in experiments], dtype=float)
    bias = np.asarray([item["bias"] for item in experiments], dtype=float)
    dimensionless = _homogeneous_scgf(field, bias)
    unstable = field * field - (field + bias) ** 2 > _TWO_PI_SQUARED + 1.0e-10
    for i in np.flatnonzero(unstable):
        dimensionless[i] = _modulated_scgf(float(field[i]), float(bias[i]))
    return float(diffusivity) * dimensionless


class Model:
    def __init__(self):
        self.diffusivity = 1.0

    def fit(self, records):
        """Fit the sole unknown parameter by inverse-variance least squares."""
        records = list(records)
        if not records:
            return self
        inputs = [record["input"] for record in records]
        response = np.asarray([record["value"] for record in records], dtype=float)
        sigma = np.asarray([record["sigma"] for record in records], dtype=float)
        shape = predict_at(inputs, 1.0)
        valid = np.isfinite(response) & np.isfinite(sigma) & (sigma > 0.0) & np.isfinite(shape)
        if np.any(valid):
            weight = 1.0 / np.square(sigma[valid])
            denominator = float(np.sum(weight * np.square(shape[valid])))
            if denominator > 0.0 and np.isfinite(denominator):
                estimate = float(np.sum(weight * shape[valid] * response[valid]) / denominator)
                if np.isfinite(estimate):
                    self.diffusivity = float(np.clip(estimate, 0.8, 1.2))
        return self

    def predict(self, experiments):
        return predict_at(experiments, self.diffusivity)
