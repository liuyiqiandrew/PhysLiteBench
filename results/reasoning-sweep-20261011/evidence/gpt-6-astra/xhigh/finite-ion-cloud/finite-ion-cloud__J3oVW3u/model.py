"""Canonical osmotic pressure for one, two, or three ions in a finite cell.

With L = log(R/a), phi = (1 - exp(-2L)) * d(log Z)/dL / (2N).
All lengths in Z are scaled by a, so the disk radius drops out of phi.
The radial integrals below use ordered log radii; constants independent
of L (including particle permutations and overall rotations) cancel.
"""

from functools import lru_cache

import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize_scalar
from scipy.special import exprel, hyp2f1, roots_legendre


@lru_cache(maxsize=4096)
def coefficient(gamma, count, ratio):
    """Return the finite-cell coefficient at coupling Gamma = u*g."""
    if count == 3:
        return three_ion_coefficient(gamma, ratio)

    length = np.log(ratio)
    area_factor = -np.expm1(-2 * length)
    if count == 1:
        slope = 2 - gamma
        # exprel also evaluates the removable singularity at Gamma = 2.
        return area_factor / (2 * length * exprel(-slope * length))
    if count != 2:
        raise ValueError("ion_count must be 1, 2, or 3")

    slope, gap_slope = 4 - 3 * gamma, 2 - gamma

    def angular(gap):
        return hyp2f1(-gamma / 2, -gamma / 2, 1, np.exp(-2 * gap))

    # Integrate the innermost log radius analytically, leaving the gap.
    partition = quad(
        lambda gap: angular(gap) * np.exp(gap_slope * gap)
        * (length - gap) * exprel(slope * (length - gap)),
        0, length, epsabs=1e-11, epsrel=1e-11,
    )[0]
    boundary = quad(
        lambda gap: angular(gap)
        * np.exp(gap_slope * gap + slope * (length - gap)),
        0, length, epsabs=1e-11, epsrel=1e-11,
    )[0]
    return area_factor * boundary / (4 * partition)


def predict_at(experiments, coupling):
    return np.asarray([
        coefficient(e["temperature_multiplier"] * coupling,
                    e["ion_count"], e["radius_ratio"])
        for e in experiments
    ], dtype=float)


class Model:
    def __init__(self):
        self.coupling = None

    def fit(self, records):
        """Fit the shared g by bounded Gaussian maximum likelihood."""
        records = list(records)
        if not records:
            raise ValueError("at least one calibration record is required")
        experiments = [r["input"] for r in records]
        values = np.asarray([r["value"] for r in records], dtype=float)
        sigmas = np.asarray([r["sigma"] for r in records], dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("calibration values must be finite")
        if not np.isfinite(sigmas).all() or np.any(sigmas <= 0):
            raise ValueError("calibration sigma must be finite and positive")

        def objective(coupling):
            residual = (predict_at(experiments, coupling) - values) / sigmas
            return np.dot(residual, residual)

        result = minimize_scalar(
            objective, bounds=(0.7, 1.4), method="bounded",
            options={"xatol": 1e-12},
        )
        if not result.success:
            raise RuntimeError("coupling fit failed: " + result.message)
        # Include the exact endpoints for a constrained optimum there.
        self.coupling = float(min((0.7, result.x, 1.4), key=objective))
        return self

    def predict(self, experiments):
        if self.coupling is None:
            raise ValueError("fit the model before predicting")
        return predict_at(experiments, self.coupling)


@lru_cache(maxsize=16)
def nodes(order):
    x, w = roots_legendre(order)
    return (x + 1) / 2, w / 2


def _three_ion_angular(gamma, gap1, gap2, modes=32):
    """Joint angular average of the three pair Boltzmann factors.

    For p = Gamma/2, the m-th Fourier coefficient of
    |1 - q*exp(i*theta)|**Gamma is

        c_m(q) = q**m * (-p)_m/m! * 2F1(-p, m-p; m+1; q**2).

    Integrating the two relative angles gives
    c_0(q1)c_0(q2)c_0(q1*q2) + 2*sum_{m>=1} c_m(q1)c_m(q2)c_m(q1*q2).
    The nonzero modes retain the correlations around the triangle.
    Keeping 32 modes resolves the integrated coefficient to about 1e-9
    over the stated domain; integer p terminates the sum exactly.
    """
    p = gamma / 2
    gaps = np.asarray([gap1, gap2, gap1 + gap2])
    squared_ratios = np.exp(-2 * gaps)
    angular = np.prod(hyp2f1(-p, -p, 1, squared_ratios), axis=0)
    prefactor = 1.0
    for mode in range(1, modes + 1):
        prefactor *= (mode - 1 - p) / mode
        if prefactor == 0:
            break
        coefficients = (
            prefactor * np.exp(-mode * gaps)
            * hyp2f1(-p, mode - p, mode + 1, squared_ratios)
        )
        angular += 2 * np.prod(coefficients, axis=0)
    return angular


def three_ion_coefficient(gamma, ratio, order=32):
    """Integrate the two radial gaps, with the common radius analytic."""
    length = np.log(ratio)
    x, w = nodes(order)
    gap1 = np.broadcast_to((length * x)[:, None], (order, order)).ravel()
    gap2 = ((length - length * x)[:, None] * x[None, :]).ravel()
    weights = (
        length * (length - length * x)[:, None] * w[:, None] * w[None, :]
    ).ravel()
    angular = _three_ion_angular(gamma, gap1, gap2)
    span = length - gap1 - gap2
    slope = 6 - 6 * gamma
    weight = weights * angular * np.exp(
        (4 - 3 * gamma) * gap1 + (2 - gamma) * gap2
    )
    # The common log radius runs from zero to span. Differentiating Z
    # with respect to L fixes the outermost ion at the hard outer wall.
    partition = np.dot(weight, span * exprel(slope * span))
    boundary = np.dot(weight, np.exp(slope * span))
    return -np.expm1(-2 * length) * boundary / (6 * partition)
