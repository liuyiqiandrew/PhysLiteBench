"""Large-ring current SCGF for the half-filled weakly asymmetric process.

Time is microscopic, so the entire SCGF is proportional to diffusivity.
The homogeneous solution loses stability when lambda*(lambda + 2*E)
falls below -4*pi**2. At half filling the ensuing optimal density wave
has zero velocity; its stationary profile is expressible with elliptic
functions.
"""
from functools import lru_cache

import numpy as np
from scipy.optimize import brentq
from scipy.special import ellipe, ellipk, elliprj


@lru_cache(maxsize=256)
def _zero_current_modulus(field):
    """Elliptic parameter at lambda = -E, where 4*K(m) = E."""
    return brentq(
        lambda m: 4.0 * ellipk(m) - field,
        0.0, np.nextafter(1.0, 0.0), xtol=5e-15,
    )


@lru_cache(maxsize=4096)
def _unit_scgf(field, bias):
    flat = bias * (bias + 2.0 * field) / 4.0
    # Within rounding error of the bifurcation, the branches agree to
    # quadratic order. Avoid an ill-conditioned, vanishing root interval.
    if flat >= -np.pi**2 - 64.0 * np.finfo(float).eps * np.pi**2:
        return flat

    # Write the density as rho = (1 + u)/2. Eliminating the constant
    # current from the hydrodynamic variational principle gives
    #   psi/D = max_u [s^2/(4*A) - E^2*<1-u^2>/4
    #                            - <u'^2/(1-u^2)>/4],
    # with s=lambda+E and A=<1/(1-u^2)> on a unit periodic ring.
    # The half-filled optimum is u=a*sn(4*K(m)*x | m), with
    # b=4*K(m)/E, a^2=m*b^2, and
    #   s^2 = E^2*(1-a^2)*(1-b^2)*(Pi(a^2,m)/K(m))^2.
    # Solve this scalar equation on the fundamental, single-wave branch.
    s_squared = (bias + field)**2
    upper = _zero_current_modulus(field)

    if s_squared == 0.0:
        modulus = upper
    else:
        def equation(m):
            if m == upper:
                return -s_squared
            k = ellipk(m)
            b_squared = (4.0 * k / field)**2
            a_squared = m * b_squared
            # Complete third-kind integral via Carlson's symmetric RJ.
            average = 1.0 + a_squared * elliprj(
                0.0, 1.0 - m, 1.0, 1.0 - a_squared
            ) / (3.0 * k)
            return (field**2 * (1.0 - a_squared)
                    * (1.0 - b_squared) * average**2 - s_squared)

        modulus = brentq(equation, 0.0, upper, xtol=5e-15)

    k = ellipk(modulus)
    # Substituting the profile into the functional cancels the explicit
    # current terms. This expression also joins the flat branch smoothly.
    return 4.0 * k**2 * (1.0 - modulus) - 8.0 * k * ellipe(modulus)


def predict_at(experiments, diffusivity):
    """Return raw SCGF values, including the nonuniform low-current phase."""
    return float(diffusivity) * np.asarray([
        _unit_scgf(float(e["field"]), float(e["bias"]))
        for e in experiments
    ], dtype=float)


class Model:
    def __init__(self):
        self.diffusivity = 1.

    def fit(self, records):
        """Bounded Gaussian maximum likelihood with known measurement errors."""
        records = list(records)
        if not records:
            return self
        basis = predict_at([r["input"] for r in records], 1.0)
        values = np.asarray([r["value"] for r in records], dtype=float)
        sigma = np.asarray([r["sigma"] for r in records], dtype=float)
        if (not np.isfinite(values).all() or not np.isfinite(sigma).all()
                or np.any(sigma <= 0.0)):
            raise ValueError("Calibration values must be finite and sigma positive.")
        weighted_basis = basis / sigma
        information = weighted_basis @ weighted_basis
        if information > 0.0:
            estimate = (weighted_basis @ (values / sigma)) / information
            self.diffusivity = float(np.clip(estimate, 0.8, 1.2))
        return self

    def predict(self, experiments):
        return predict_at(experiments, self.diffusivity)
