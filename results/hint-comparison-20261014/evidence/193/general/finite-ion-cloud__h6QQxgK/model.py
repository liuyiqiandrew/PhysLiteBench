"""Finite-cell osmotic coefficient for the two-dimensional ionic cell.

The radial coordinate is represented by ``t = log(r/a)``.  Sorting the
radial coordinates leaves only one angular integral per ion pair,

    <|1 - exp(-d+i theta)|**gamma> =
        2F1(-gamma/2, -gamma/2; 1; exp(-2 d)).

This makes the one- and two-ion answers one-dimensional quadratures and the
three-ion answer a smooth two-dimensional quadrature.  ``disk_radius`` does
not occur below: at fixed R/a it is an overall configurational scale and
cancels from the pressure ratio.
"""

from functools import lru_cache

import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize_scalar
from scipy.special import exprel, hyp2f1, roots_legendre


def _pair_angular_factor(gamma, gap):
    """Angular average of a pair factor, for a nonnegative radial gap."""
    return hyp2f1(-gamma / 2.0, -gamma / 2.0, 1.0, np.exp(-2.0 * gap))


@lru_cache(maxsize=None)
def coefficient(gamma, count, ratio):
    """Return the finite-cell osmotic coefficient at ``Gamma=gamma``.

    The pressure follows from differentiating log of the configurational
    integral with respect to L=log(R/a).  The common factor is
    (1-exp(-2L))/(2N); the ratios below are the remaining log derivatives.
    """
    gamma = float(gamma)
    count = int(count)
    ratio = float(ratio)
    if count not in (1, 2, 3):
        raise ValueError("ion_count must be 1, 2, or 3")
    if not (np.isfinite(gamma) and np.isfinite(ratio) and ratio > 1.0):
        raise ValueError("gamma must be finite and radius_ratio must exceed 1")

    length = np.log(ratio)
    area_factor = -np.expm1(-2.0 * length)

    if count == 1:
        slope = 2.0 - gamma
        partition = length * exprel(slope * length)
        boundary = np.exp(slope * length)
        return float(area_factor * boundary / (2.0 * partition))

    if count == 2:
        # d is the radial log-gap and s is the lower radial coordinate.
        lower_slope = 4.0 - 3.0 * gamma
        gap_slope = 2.0 - gamma

        def integrand(gap):
            span = length - gap
            return (_pair_angular_factor(gamma, gap) * np.exp(gap_slope * gap)
                    * span * exprel(lower_slope * span))

        def boundary_integrand(gap):
            span = length - gap
            return (_pair_angular_factor(gamma, gap) * np.exp(
                gap_slope * gap + lower_slope * span))

        partition = quad(integrand, 0.0, length, epsabs=1e-11,
                         epsrel=1e-11, limit=100)[0]
        boundary = quad(boundary_integrand, 0.0, length, epsabs=1e-11,
                        epsrel=1e-11, limit=100)[0]
        return float(area_factor * boundary / (4.0 * partition))

    return three_ion_coefficient(gamma, ratio)


def predict_at(experiments, coupling):
    """Evaluate inputs in their supplied order."""
    return np.asarray([
        coefficient(float(experiment["temperature_multiplier"]) * coupling,
                    experiment["ion_count"], experiment["radius_ratio"])
        for experiment in experiments
    ], dtype=float)


class Model:
    def __init__(self):
        self.coupling = None

    def fit(self, records):
        """Fit the shared reference coupling by Gaussian weighted least squares."""
        records = list(records)
        if not records:
            raise ValueError("at least one calibration record is required")

        experiments = [record["input"] for record in records]
        values = np.asarray([record["value"] for record in records], dtype=float)
        sigmas = np.asarray([record["sigma"] for record in records], dtype=float)
        if (not np.isfinite(values).all() or not np.isfinite(sigmas).all()
                or np.any(sigmas <= 0.0)):
            raise ValueError("calibration values must be finite and sigmas positive")

        def objective(coupling):
            residuals = (predict_at(experiments, coupling) - values) / sigmas
            return float(np.dot(residuals, residuals))

        result = minimize_scalar(objective, bounds=(0.7, 1.4), method="bounded",
                                 options={"xatol": 1e-11})
        if not result.success or not np.isfinite(result.fun):
            raise RuntimeError("unable to fit electrostatic coupling")
        self.coupling = float(result.x)
        return self

    def predict(self, experiments):
        if self.coupling is None:
            raise RuntimeError("fit must be called before predict")
        return predict_at(experiments, self.coupling)


@lru_cache(maxsize=None)
def nodes(order):
    x, w = roots_legendre(order)
    return (x + 1.0) / 2.0, w / 2.0


def three_ion_coefficient(gamma, ratio, order=32):
    """Three-ion result from tensor Gauss--Legendre integration.

    ``d1`` and ``d2`` are the two adjacent radial log-gaps and ``s`` is the
    lowest coordinate.  The final s integration is analytic.  Order 32 is
    converged to substantially better than 1e-7 over the specified domain.
    """
    length = np.log(ratio)
    x, w = nodes(order)
    d1 = np.broadcast_to((length * x)[:, None], (order, order)).ravel()
    d2 = ((length - length * x)[:, None] * x[None, :]).ravel()
    weights = (length * (length - length * x)[:, None]
               * w[:, None] * w[None, :]).ravel()

    angular = (_pair_angular_factor(gamma, d1)
               * _pair_angular_factor(gamma, d2)
               * _pair_angular_factor(gamma, d1 + d2))
    span = length - d1 - d2
    lower_slope = 6.0 - 6.0 * gamma

    # Interchanging d1 and d2 gives the same triangular integral.  This
    # version assigns their slopes to the upper and lower adjacent gaps.
    radial_weight = weights * angular * np.exp(
        (2.0 - gamma) * d1 + (4.0 - 3.0 * gamma) * d2)
    partition = np.dot(radial_weight, span * exprel(lower_slope * span))
    boundary = np.dot(radial_weight, np.exp(lower_slope * span))
    area_factor = -np.expm1(-2.0 * length)
    return float(area_factor * boundary / (6.0 * partition))
