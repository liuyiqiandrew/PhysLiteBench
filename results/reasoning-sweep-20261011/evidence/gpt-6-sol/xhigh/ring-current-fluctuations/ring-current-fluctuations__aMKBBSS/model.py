"""Thermodynamic current SCGF for the half-filled weakly asymmetric ring."""

from functools import lru_cache

import numpy as np
from scipy.optimize import minimize


_WAVE_NUMBER = 2.0 * np.pi
_CRITICAL_FIELD_SQUARED = _WAVE_NUMBER**2
_ODD_MODES = np.arange(1, 24, 2)
_GRID = (np.arange(192) + 0.5) / 192
_COSINES = np.cos(2.0 * np.pi * _GRID[:, None] * _ODD_MODES)
_FREQUENCIES_SQUARED = (2.0 * np.pi * _ODD_MODES) ** 2


@lru_cache(maxsize=4096)
def _unit_scgf(field, bias):
    """SCGF with microscopic rate prefactor D=1.

    In the uniform phase the answer is lambda(lambda+2E)/4. Below the
    stability threshold, the optimal half-filled density is a stationary
    wave. Write its density as (1+sin(theta(x)))/2 and represent theta by
    odd cosine modes, which enforce the half-filling symmetry. The resulting
    one-dimensional MFT variational problem is maximized numerically.
    """
    e = float(field)
    k = float(bias) + e
    flat = (k * k - e * e) / 4.0
    if e * e - k * k <= _CRITICAL_FIELD_SQUARED:
        return flat

    def objective(coefficients):
        theta = _COSINES @ coefficients
        sine = np.sin(theta)
        cosine = np.cos(theta)
        secant_squared = 1.0 / (cosine * cosine)
        resistance = np.mean(secant_squared)
        density_variance = np.mean(sine * sine)
        gradient_cost = np.dot(_FREQUENCIES_SQUARED, coefficients**2) / 2.0

        # Four times the SCGF equals the following maximum minus E^2.
        value = k * k / resistance + e * e * density_variance - gradient_cost
        local_gradient = (
            e * e * np.sin(2.0 * theta)
            - 2.0 * k * k / resistance**2 * secant_squared * sine / cosine
        )
        gradient = (
            _COSINES.T @ local_gradient / len(_GRID)
            - _FREQUENCIES_SQUARED * coefficients
        )
        return -value, -gradient

    initial = np.zeros(len(_ODD_MODES))
    initial[0] = 0.7
    result = minimize(
        objective,
        initial,
        jac=True,
        method="BFGS",
        options={"gtol": 1e-9, "maxiter": 300},
    )
    return max(flat, (-result.fun - e * e) / 4.0)


def predict_at(experiments, diffusivity):
    """Return the asymptotic SCGF at each input, in microscopic time units."""
    return float(diffusivity) * np.asarray(
        [_unit_scgf(item["field"], item["bias"]) for item in experiments],
        dtype=float,
    )


class Model:
    def __init__(self):
        self.diffusivity = 1.0

    def fit(self, records):
        inputs = [record["input"] for record in records]
        if inputs:
            unit_values = predict_at(inputs, 1.0)
            values = np.asarray([record["value"] for record in records], dtype=float)
            sigmas = np.asarray([record["sigma"] for record in records], dtype=float)
            scaled_model = unit_values / sigmas
            scaled_values = values / sigmas
            information = np.dot(scaled_model, scaled_model)
            if information > 0:
                estimate = np.dot(scaled_model, scaled_values) / information
                self.diffusivity = float(np.clip(estimate, 0.8, 1.2))
        return self

    def predict(self, experiments):
        return predict_at(experiments, self.diffusivity)
