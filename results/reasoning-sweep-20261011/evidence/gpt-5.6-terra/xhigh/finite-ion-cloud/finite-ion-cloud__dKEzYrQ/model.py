"""Finite-cell osmotic coefficients for the charged-disk apparatus.

The length scale ``a`` drops out after writing every radial coordinate as
``r/a``. Thus, although it remains part of the public experiment input,
``disk_radius`` must not affect the result.
"""

from functools import lru_cache

import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize_scalar
from scipy.special import exprel, hyp2f1, roots_legendre


def _check_parameters(gamma, count, ratio):
    if count not in (1, 2, 3):
        raise ValueError("ion_count must be 1, 2, or 3")
    if not (np.isfinite(gamma) and gamma >= 0.0):
        raise ValueError("the electrostatic coupling must be finite and non-negative")
    if not (np.isfinite(ratio) and ratio > 1.0):
        raise ValueError("radius_ratio must be finite and greater than one")


@lru_cache(maxsize=None)
def coefficient(gamma, count, ratio):
    """Return the canonical osmotic coefficient at fixed ``Gamma``.

    The radial coordinates are ordered before integrating. Constants from the
    ordering and from the absolute rotation cancel in ``d log(Z)/dR``.
    """
    gamma = float(gamma)
    count = int(count)
    ratio = float(ratio)
    _check_parameters(gamma, count, ratio)

    length = np.log(ratio)
    outer_factor = -np.expm1(-2.0 * length)

    if count == 1:
        slope = 2.0 - gamma
        # Z is proportional to L * exprel(slope * L).
        return outer_factor * np.exp(slope * length) / (
            2.0 * length * exprel(slope * length)
        )

    if count == 2:
        # d is the log-radius gap, and s is the smaller log-radius.
        radial_slope, gap_slope = 4.0 - 3.0 * gamma, 2.0 - gamma

        def pair_angle_average(gap):
            # Mean over the one relative polar angle.
            return hyp2f1(-gamma / 2.0, -gamma / 2.0, 1.0, np.exp(-2.0 * gap))

        partition = quad(
            lambda gap: pair_angle_average(gap)
            * np.exp(gap_slope * gap)
            * (length - gap)
            * exprel(radial_slope * (length - gap)),
            0.0,
            length,
            epsabs=1e-11,
            epsrel=1e-11,
            limit=100,
        )[0]
        boundary = quad(
            lambda gap: pair_angle_average(gap)
            * np.exp(gap_slope * gap + radial_slope * (length - gap)),
            0.0,
            length,
            epsabs=1e-11,
            epsrel=1e-11,
            limit=100,
        )[0]
        return outer_factor * boundary / (4.0 * partition)

    return three_ion_coefficient(gamma, ratio)


def predict_at(experiments, coupling):
    """Evaluate an iterable of public experiment dictionaries."""
    coupling = float(coupling)
    return np.asarray(
        [
            coefficient(
                float(experiment["temperature_multiplier"]) * coupling,
                int(experiment["ion_count"]),
                float(experiment["radius_ratio"]),
            )
            for experiment in experiments
        ],
        dtype=float,
    )


@lru_cache(maxsize=None)
def nodes(order):
    """Gauss--Legendre nodes and weights mapped to [0, 1]."""
    x, weights = roots_legendre(order)
    return (x + 1.0) / 2.0, weights / 2.0


def _fourier_prefactors(half_gamma, modes):
    """Coefficients of ``(1-q exp(i theta))**half_gamma``."""
    values = np.empty(modes + 1, dtype=float)
    values[0] = 1.0
    for index in range(1, modes + 1):
        # This recurrence remains well behaved at integer half-couplings,
        # where direct gamma-function formulas have poles and zeros.
        values[index] = values[index - 1] * (index - 1.0 - half_gamma) / index
    return values


def _three_body_angle_average(gamma, inner_gap, outer_gap, modes):
    """Average all three pair factors over the two relative angles.

    The three pair angles are correlated. If c_n(q) are the Fourier
    coefficients of ``(1 - 2q cos(theta) + q**2)**(Gamma/2)``, the required
    two-angle average is ``c0(q1)c0(q2)c0(q1*q2) +
    2*sum(cn(q1)cn(q2)cn(q1*q2), n>0)``. This exact convolution is more
    accurate than treating the three pair-angle averages as independent.
    """
    half_gamma = gamma / 2.0
    index = np.arange(modes + 1, dtype=float)
    prefactor = _fourier_prefactors(half_gamma, modes)

    def coefficients(gap):
        # c_n = (-p)_n/n! q^n 2F1(n-p, -p; n+1; q^2), p=Gamma/2.
        gap = np.asarray(gap, dtype=float)[:, None]
        return (
            prefactor[None, :]
            * np.exp(-gap * index[None, :])
            * hyp2f1(
                index[None, :] - half_gamma,
                -half_gamma,
                index[None, :] + 1.0,
                np.exp(-2.0 * gap),
            )
        )

    first = coefficients(inner_gap)
    second = coefficients(outer_gap)
    third = coefficients(np.asarray(inner_gap) + np.asarray(outer_gap))
    terms = first * second * third
    return terms[:, 0] + 2.0 * np.sum(terms[:, 1:], axis=1)


def three_ion_coefficient(gamma, ratio, order=32, angular_modes=32):
    """Canonical three-ion coefficient using radial and angular quadrature.

    The default orders converge far beyond the calibration precision over the
    stated domain. Optional orders make independent convergence checks easy.
    """
    gamma = float(gamma)
    ratio = float(ratio)
    _check_parameters(gamma, 3, ratio)
    if order < 2 or angular_modes < 1:
        raise ValueError("quadrature orders must be positive")

    order, angular_modes = int(order), int(angular_modes)
    length = np.log(ratio)
    x, weights = nodes(order)

    # Duffy coordinates for the triangle of non-negative radial gaps.
    # inner_gap is t_middle-t_inner; outer_gap is t_outer-t_middle.
    inner_gap = np.broadcast_to((length * x)[:, None], (order, order)).ravel()
    outer_gap = ((length - length * x)[:, None] * x[None, :]).ravel()
    integration_weights = (
        length * (length - length * x)[:, None] * weights[:, None] * weights[None, :]
    ).ravel()

    angle_average = _three_body_angle_average(
        gamma, inner_gap, outer_gap, angular_modes
    )
    inner_radius_span = length - inner_gap - outer_gap
    radial_slope = 6.0 - 6.0 * gamma
    gap_weight = np.exp(
        (4.0 - 3.0 * gamma) * inner_gap + (2.0 - gamma) * outer_gap
    )
    weight = integration_weights * angle_average * gap_weight
    partition = np.dot(
        weight, inner_radius_span * exprel(radial_slope * inner_radius_span)
    )
    boundary = np.dot(weight, np.exp(radial_slope * inner_radius_span))

    outer_factor = -np.expm1(-2.0 * length)
    return outer_factor * boundary / (6.0 * partition)


class Model:
    def __init__(self):
        self.coupling = None

    def fit(self, records):
        """Fit the shared reference coupling by weighted least squares."""
        records = list(records)
        if not records:
            raise ValueError("at least one calibration record is required")

        experiments = [record["input"] for record in records]
        observed = np.asarray([record["value"] for record in records], dtype=float)
        sigma = np.asarray([record["sigma"] for record in records], dtype=float)
        if not (
            np.isfinite(observed).all()
            and np.isfinite(sigma).all()
            and (sigma > 0).all()
        ):
            raise ValueError("calibration values must be finite and sigma must be positive")

        def objective(coupling):
            residual = (predict_at(experiments, coupling) - observed) / sigma
            return float(np.dot(residual, residual))

        result = minimize_scalar(
            objective,
            bounds=(0.7, 1.4),
            method="bounded",
            options={"xatol": 1e-10},
        )
        if not result.success or not np.isfinite(result.fun):
            raise RuntimeError("could not fit the electrostatic coupling")
        self.coupling = float(result.x)
        return self

    def predict(self, experiments):
        if self.coupling is None:
            raise RuntimeError("fit must be called before predict")
        prediction = predict_at(experiments, self.coupling)
        if not np.isfinite(prediction).all():
            raise FloatingPointError("non-finite osmotic-coefficient prediction")
        return prediction
