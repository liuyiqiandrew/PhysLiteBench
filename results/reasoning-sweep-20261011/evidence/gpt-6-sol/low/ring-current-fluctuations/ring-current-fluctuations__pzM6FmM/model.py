"""Asymptotic current SCGF for the half-filled weakly asymmetric ring."""

from functools import lru_cache

import numpy as np
from scipy.optimize import minimize


@lru_cache(maxsize=1024)
def _unit_scgf(field, bias):
    """SCGF at unit microscopic rate, including the inactive wave phase."""
    e, lam = float(field), float(bias)
    uniform = lam * (lam + 2.0 * e) / 4.0
    if lam * (lam + 2.0 * e) >= -4.0 * np.pi**2:
        return uniform

    # Particle-hole symmetry at half filling makes the optimal wave stationary.
    # Write rho=(1+sin(theta))/2. Fourier modes with odd wave numbers encode
    # the two symmetric halves of the density profile without a singularity at
    # rho=0 or 1. The functional below is the current-optimized MFT action.
    modes = 12
    x = (np.arange(512) + 0.5) / 512.0
    wave = 2.0 * np.pi * np.arange(1, 2 * modes, 2)
    basis = np.cos(wave[:, None] * x[None, :])
    derivative = -wave[:, None] * np.sin(wave[:, None] * x[None, :])
    drive = (lam + e) ** 2

    def objective(coeff):
        theta = coeff @ basis
        if np.max(np.abs(theta)) >= 1.56:
            return 1e4 + 1e4 * (np.max(np.abs(theta)) - 1.56) ** 2
        cos2 = np.cos(theta) ** 2
        inv = 1.0 / cos2
        grad = coeff @ derivative
        score = drive / (4.0 * np.mean(inv)) - e * e * np.mean(cos2) / 4.0 - np.mean(grad * grad) / 4.0
        return -score

    start = np.zeros(modes)
    start[0] = 0.7
    result = minimize(objective, start, method="BFGS", options={"gtol": 1e-8, "maxiter": 1000})
    return max(uniform, -float(result.fun))


def predict_at(experiments, diffusivity):
    """Predict raw SCGF values in inverse microscopic time units."""
    return np.asarray(
        [diffusivity * _unit_scgf(item["field"], item["bias"]) for item in experiments],
        dtype=float,
    )


class Model:
    def __init__(self):
        self.diffusivity = 1.0

    def fit(self, records):
        if not records:
            return self
        base = predict_at([record["input"] for record in records], 1.0)
        values = np.asarray([record["value"] for record in records], dtype=float)
        sigma = np.asarray([record["sigma"] for record in records], dtype=float)
        weight = 1.0 / sigma**2
        denominator = np.dot(weight, base * base)
        if denominator > 0:
            estimate = np.dot(weight, base * values) / denominator
            self.diffusivity = float(np.clip(estimate, 0.8, 1.2))
        return self

    def predict(self, experiments):
        return predict_at(experiments, self.diffusivity)
