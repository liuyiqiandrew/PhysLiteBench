"""Large-ring current SCGF for the half-filled weakly asymmetric process.

The uniform density gives lambda * (lambda + 2 E) / 4 at unit
diffusivity.  When E**2 - (lambda + E)**2 > 4 pi**2, a nonuniform
profile instead maximizes the hydrodynamic current variational principle.
At half filling this profile has zero propagation velocity and can be
expressed using complete elliptic integrals.

Multiplying every microscopic rate by D multiplies the entire SCGF by D.
Consequently calibration is a bounded, one-parameter weighted least-squares
fit, including when calibration inputs lie in the nonuniform phase.
"""

from functools import lru_cache

import numpy as np
from scipy.optimize import brentq
from scipy.special import ellipe, ellipk, elliprj


_CRITICAL_PRODUCT = -4.0 * np.pi**2
_ROOT_TOL = 5e-15
_ROOT_RTOL = 4.0 * np.finfo(float).eps


@lru_cache(maxsize=512)
def _zero_current_parameter(field):
    """Elliptic parameter at lambda = -E, where K(m) = E / 4."""
    return brentq(
        lambda m: ellipk(m) - field / 4.0,
        0.0,
        1.0,
        xtol=_ROOT_TOL,
        rtol=_ROOT_RTOL,
    )


@lru_cache(maxsize=4096)
def _nonuniform_scgf(field, shifted_bias):
    """Unit-D SCGF on the broken-symmetry branch; shifted_bias = |E+lambda|.

Writing u = 2 rho - 1, the optimal profile satisfies
u'**2 = E**2 (a**2 - u**2) (b**2 - u**2), with m = a**2 / b**2.
Its unit spatial period fixes b = 4 K(m) / E.  Current optimization fixes

    |E + lambda| = E sqrt((1-a**2)(1-b**2)) Pi(a**2, m) / K(m).

This is a scalar equation between the uniform profile (m=0) and the
zero-current profile (b=1).  Pi is evaluated via Carlson's symmetric RJ
integral.  Substitution in the variational functional gives the final
expression below, without discretizing the ring or taking finite-L fits.
"""
    upper = _zero_current_parameter(field)

    if shifted_bias == 0.0:
        parameter = upper
    else:
        def residual(parameter):
            # Impose b=1 exactly at the endpoint: roundoff in K(upper)
            # must not spoil the bracket for biases extremely close to -E.
            if parameter == upper:
                return -shifted_bias**2
            first = ellipk(parameter)
            b_squared = (4.0 * first / field)**2
            a_squared = parameter * b_squared
            third = first + a_squared * elliprj(
                0.0, 1.0 - parameter, 1.0, 1.0 - a_squared
            ) / 3.0
            return (
                field**2 * (1.0 - a_squared) * (1.0 - b_squared)
                * (third / first)**2 - shifted_bias**2
            )

        parameter = brentq(
            residual, 0.0, upper, xtol=_ROOT_TOL, rtol=_ROOT_RTOL
        )

    first = ellipk(parameter)
    second = ellipe(parameter)
    return float(4.0 * first * ((1.0 - parameter) * first - 2.0 * second))


def predict_at(experiments, diffusivity):
    """Return raw SCGF values in inverse microscopic time units."""
    experiments = list(experiments)
    field = np.asarray([e["field"] for e in experiments], dtype=float)
    bias = np.asarray([e["bias"] for e in experiments], dtype=float)
    product = bias * (bias + 2.0 * field)
    values = product / 4.0
    # At the transition the two branches agree to first order.  Use the
    # uniform expression within roundoff of it, avoiding a degenerate root
    # bracket without sacrificing any representable prediction accuracy.
    roundoff = 32.0 * np.finfo(float).eps * np.maximum(1.0, field**2)
    for index in np.flatnonzero(product < _CRITICAL_PRODUCT - roundoff):
        values[index] = _nonuniform_scgf(
            float(field[index]), float(abs(field[index] + bias[index]))
        )
    return float(diffusivity) * values


class Model:
    def __init__(self):
        self.diffusivity = 1.0

    def fit(self, records):
        """Fit D by Gaussian maximum likelihood, constrained to [0.8, 1.2].

        Empty data, or inputs whose SCGF is identically zero, provide no
        information about D and leave its current value unchanged.
        """
        records = list(records)
        if not records:
            return self
        observed = np.asarray([r["value"] for r in records], dtype=float)
        sigma = np.asarray([r["sigma"] for r in records], dtype=float)
        if not np.all(np.isfinite(observed)):
            raise ValueError("Calibration values must be finite.")
        if not np.all(np.isfinite(sigma) & (sigma > 0.0)):
            raise ValueError("Calibration uncertainties must be finite and positive.")

        unit_values = predict_at([r["input"] for r in records], 1.0)
        # Relative weights avoid needless overflow for uniformly small errors.
        relative_precision = np.min(sigma) / sigma
        design = unit_values * relative_precision
        response = observed * relative_precision
        information = float(design @ design)
        if information > 0.0:
            estimate = float(design @ response) / information
            self.diffusivity = float(np.clip(estimate, 0.8, 1.2))
        return self

    def predict(self, experiments):
        return predict_at(experiments, self.diffusivity)
