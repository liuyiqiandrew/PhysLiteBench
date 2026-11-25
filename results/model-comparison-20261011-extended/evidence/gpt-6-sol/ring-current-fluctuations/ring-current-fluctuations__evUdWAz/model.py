"""Asymptotic current generating function for the half-filled driven ring.

At half filling the low-current optimal density is stationary and symmetric
under particle-hole exchange followed by a half-ring translation.  We
represent it by odd Fourier modes of ``theta``, where rho=(1+sin(theta))/2.
"""

from functools import lru_cache

import numpy as np
from scipy.optimize import minimize


_PI = np.pi
_WAVENUMBERS = np.arange(1, 24, 2, dtype=float)
_GRID = np.arange(384, dtype=float) / 384
_COSINES = np.cos(2 * _PI * _GRID[:, None] * _WAVENUMBERS)
_MODE_PENALTY = (_PI * _WAVENUMBERS) ** 2


def _profile_scgf(field, shifted_bias):
    """SCGF at unit diffusivity, maximized over the density profile."""
    flat = (shifted_bias * shifted_bias - field * field) / 4
    # The second variation about rho=1/2 changes sign at this boundary.
    if field * field - shifted_bias * shifted_bias <= 4 * _PI * _PI:
        return flat

    def objective(coefficients):
        theta = _COSINES @ coefficients
        if np.max(np.abs(theta)) >= _PI / 2:
            # theta must represent a density in [0, 1] without folding.
            return 1e6 + float(coefficients @ coefficients), 2 * coefficients

        cosine = np.cos(theta)
        secant_squared = 1 / (cosine * cosine)
        tangent = np.tan(theta)
        average_secant_squared = np.mean(secant_squared)
        average_cosine_squared = np.mean(cosine * cosine)

        value = (
            shifted_bias * shifted_bias / (4 * average_secant_squared)
            - np.dot(_MODE_PENALTY * coefficients, coefficients) / 2
            - field * field * average_cosine_squared / 4
        )
        derivative = (
            -shifted_bias * shifted_bias
            / (2 * average_secant_squared**2)
            * np.mean(_COSINES * (secant_squared * tangent)[:, None], axis=0)
            - _MODE_PENALTY * coefficients
            + field * field / 4
            * np.mean(_COSINES * np.sin(2 * theta)[:, None], axis=0)
        )
        return -value, -derivative

    seed = np.zeros(len(_WAVENUMBERS))
    seed[0] = 0.6
    result = minimize(objective, seed, method="L-BFGS-B", jac=True,
                      options={"ftol": 2e-14, "gtol": 1e-10, "maxiter": 400})
    return max(flat, -float(result.fun))


@lru_cache(maxsize=4096)
def _unit_scgf(field, bias):
    return _profile_scgf(field, abs(field + bias))


def predict_at(experiments, diffusivity):
    """Predict the raw SCGF for a sequence of field/bias dictionaries."""
    return np.asarray(
        [diffusivity * _unit_scgf(float(item["field"]), float(item["bias"]))
         for item in experiments], dtype=float
    )


class Model:
    def __init__(self):
        self.diffusivity = 1.0

    def fit(self, records):
        """Weighted maximum likelihood estimate for the common rate scale."""
        if not records:
            raise ValueError("At least one calibration record is required")
        predictions = predict_at([record["input"] for record in records], 1.0)
        values = np.asarray([record["value"] for record in records], dtype=float)
        sigma = np.asarray([record["sigma"] for record in records], dtype=float)
        weights = 1 / (sigma * sigma)
        denominator = np.dot(weights * predictions, predictions)
        if denominator <= 0 or not np.isfinite(denominator):
            raise ValueError("Calibration records do not identify diffusivity")
        estimate = np.dot(weights * predictions, values) / denominator
        self.diffusivity = float(np.clip(estimate, 0.8, 1.2))
        return self

    def predict(self, inputs):
        return predict_at(inputs, self.diffusivity)
