"""Macroscopic-fluctuation-theory prediction for the weakly asymmetric ring.

The rates in the question give diffusion coefficient ``D`` and mobility
``2 D rho (1-rho)``.  At half filling the large-ring SCGF is ``D`` times a
dimensionless variational problem.  Keeping that factor outside is useful:
the shape of a conditioned density wave does not depend on the fitted value
of ``D``.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import LinearConstraint, minimize


# Numerical resolution for the one-dimensional smooth variational profile.
# The Fourier representation fixes the arbitrary translation of a density
# wave by putting its maximum at x=0.
_MODES = 16
_POINTS = 96
_EPS = 1.0e-7


def _homogeneous_scgf(field: float, bias: float) -> float:
    """SCGF of the spatially homogeneous saddle, with D factored out."""
    return bias * (bias + 2.0 * field) / 4.0


def _is_homogeneous_stable(field: float, bias: float) -> bool:
    """Long-wavelength stability condition of the half-filled saddle."""
    # The first unstable Fourier mode has wave number 2*pi.  Writing the
    # condition this way avoids an unnecessary nonlinear solve throughout
    # the (large) homogeneous part of the domain.
    return field <= 2.0 * np.pi or (field + bias) ** 2 >= field**2 - 4.0 * np.pi**2


def _modulated_scgf(field: float, bias: float) -> float:
    """Solve the modulated, half-filled MFT saddle.

    For a travelling density profile the wave speed vanishes at density 1/2
    by particle-hole symmetry.  On eliminating its constant current, the
    remaining variational problem is

        max_rho (E+l)^2/(2 <1/m>) - E^2 <m>/2 - <(rho')^2/m>/2,
        m = 2 rho (1-rho),  <rho> = 1/2.

    This is the branch missed by a homogeneous-density calculation.  A
    cosine expansion is sufficient because a one-dimensional stationary
    orbit can be translated to be reflection symmetric.
    """
    n = np.arange(1, _MODES + 1, dtype=float)
    x = np.arange(_POINTS, dtype=float) / _POINTS
    cosines = np.cos(2.0 * np.pi * np.outer(x, n))
    derivatives = -2.0 * np.pi * np.sin(2.0 * np.pi * np.outer(x, n)) * n

    def negative_value(coefficients: np.ndarray) -> float:
        rho = 0.5 + cosines @ coefficients
        rho_x = derivatives @ coefficients
        mobility = 2.0 * rho * (1.0 - rho)
        inverse_mobility = np.mean(1.0 / mobility)
        value = (
            (field + bias) ** 2 / (2.0 * inverse_mobility)
            - 0.5 * field**2 * np.mean(mobility)
            - 0.5 * np.mean(rho_x**2 / mobility)
        )
        return -float(value)

    # Pointwise constraints keep the mobility nonsingular.  Several initial
    # amplitudes make the result robust both close to the continuous
    # transition and deep in the wave phase.
    constraint = LinearConstraint(cosines, _EPS - 0.5, 0.5 - _EPS)
    best_value = _homogeneous_scgf(field, bias)
    for amplitude in (0.01, 0.08, 0.20, 0.40, 0.55):
        initial = np.zeros(_MODES)
        initial[0] = amplitude
        result = minimize(
            negative_value,
            initial,
            method="SLSQP",
            bounds=[(-0.7, 0.7)] * _MODES,
            constraints=(constraint,),
            options={"ftol": 3.0e-12, "maxiter": 1200},
        )
        if result.success and np.isfinite(result.fun):
            best_value = max(best_value, -float(result.fun))
    return best_value


def _unit_diffusivity_scgf(field: float, bias: float) -> float:
    """Return psi/D, including the density-wave branch when it is selected."""
    homogeneous = _homogeneous_scgf(field, bias)
    if _is_homogeneous_stable(field, bias):
        return homogeneous
    return _modulated_scgf(field, bias)


def predict_at(experiments, diffusivity):
    """Predict raw SCGF values for experiment dictionaries."""
    if not experiments:
        return np.empty(0, dtype=float)

    values = np.empty(len(experiments), dtype=float)
    # A small per-call cache matters when predictions contain repeated input
    # records and leaves no assumptions about ordering.
    cached = {}
    for index, experiment in enumerate(experiments):
        field = float(experiment["field"])
        bias = float(experiment["bias"])
        key = (field, bias)
        if key not in cached:
            cached[key] = _unit_diffusivity_scgf(field, bias)
        values[index] = diffusivity * cached[key]
    return values


class Model:
    def __init__(self):
        self.diffusivity = 1.0

    def fit(self, records):
        """Fit the common clock-rate prefactor by weighted least squares."""
        if not records:
            # There is no information with which to update the documented
            # finite default.  Returning self preserves the normal API.
            return self

        unit_values = np.array(
            [
                _unit_diffusivity_scgf(float(r["input"]["field"]), float(r["input"]["bias"]))
                for r in records
            ],
            dtype=float,
        )
        observations = np.array([r["value"] for r in records], dtype=float)
        errors = np.array([r["sigma"] for r in records], dtype=float)
        weights = 1.0 / errors**2
        denominator = float(np.sum(weights * unit_values**2))
        if np.isfinite(denominator) and denominator > 0.0:
            estimate = float(np.sum(weights * unit_values * observations) / denominator)
            self.diffusivity = float(np.clip(estimate, 0.8, 1.2))
        return self

    def predict(self, experiments):
        return predict_at(experiments, self.diffusivity)
