"""Canonical osmotic coefficient for the finite two-dimensional cell.

The radial co-ordinates used below are ``x = log(r/a)``. Sorting them makes
the radial Boltzmann weight simple and leaves angular averages of
``|1-q exp(i theta)|**gamma``.
"""

from functools import lru_cache

import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize_scalar
from scipy.special import exprel, hyp2f1, roots_legendre


@lru_cache(maxsize=None)
def coefficient(gamma, count, ratio):
    """Return phi at a specified dimensionless coupling."""
    length = np.log(ratio)
    if count == 1:
        slope = 2.0 - gamma
        return -np.expm1(-2.0 * length) * np.exp(slope * length) / (
            2.0 * length * exprel(slope * length)
        )
    if count == 2:
        # The two ordered log radii are x and x+d.
        common_slope, gap_slope = 4.0 - 3.0 * gamma, 2.0 - gamma

        def angular(gap):
            return hyp2f1(-gamma / 2.0, -gamma / 2.0, 1.0, np.exp(-2.0 * gap))

        partition = quad(
            lambda d: angular(d) * np.exp(gap_slope * d) * (length - d)
            * exprel(common_slope * (length - d)),
            0.0, length, epsabs=1e-11, epsrel=1e-11,
        )[0]
        boundary = quad(
            lambda d: angular(d) * np.exp(gap_slope * d + common_slope * (length - d)),
            0.0, length, epsabs=1e-11, epsrel=1e-11,
        )[0]
        return -np.expm1(-2.0 * length) * boundary / (4.0 * partition)
    if count == 3:
        return three_ion_coefficient(gamma, ratio)
    raise ValueError("ion_count must be 1, 2, or 3")


def predict_at(experiments, coupling):
    # disk_radius cancels after all distances are scaled by a.
    return np.asarray([
        coefficient(float(e["temperature_multiplier"]) * coupling,
                    int(e["ion_count"]), float(e["radius_ratio"]))
        for e in experiments
    ], dtype=float)


class Model:
    def __init__(self):
        self.coupling = None

    def fit(self, records):
        """Fit the shared reference coupling by weighted least squares."""
        records = list(records)
        if not records:
            raise ValueError("at least one calibration record is required")
        experiments = [record["input"] for record in records]
        values = np.asarray([record["value"] for record in records], dtype=float)
        sigma = np.asarray([record["sigma"] for record in records], dtype=float)
        if not np.isfinite(values).all() or not np.isfinite(sigma).all() or np.any(sigma <= 0):
            raise ValueError("calibration values must be finite and sigma must be positive")

        def loss(coupling):
            residual = (predict_at(experiments, coupling) - values) / sigma
            return float(np.dot(residual, residual))

        result = minimize_scalar(loss, bounds=(0.7, 1.4), method="bounded",
                                 options={"xatol": 1e-10})
        self.coupling = float(result.x)
        return self

    def predict(self, experiments):
        if self.coupling is None:
            raise ValueError("fit must be called before predict")
        return predict_at(experiments, self.coupling)


@lru_cache(maxsize=None)
def nodes(order):
    x, w = roots_legendre(order)
    return (x + 1.0) / 2.0, w / 2.0


@lru_cache(maxsize=None)
def _angle_grid(size):
    """sin(theta/2)**2 on an equally spaced periodic angular grid."""
    return np.sin(np.pi * np.arange(size) / size) ** 2


def _three_angle_average(gamma, first_gap, second_gap, modes=2048):
    """Joint, rather than independent-pair, three-ion angle average.

    The three pair angles are alpha, beta, and alpha+beta. Fourier expanding
    each pair factor gives sum_n c_n(q1)c_n(q2)c_n(q1*q2) after the two angle
    integrations. This enforces the triangle constraint exactly.
    """
    alpha = gamma / 2.0
    q1 = np.exp(-first_gap)[:, None]
    q2 = np.exp(-second_gap)[:, None]
    q3 = q1 * q2
    sin_squared = _angle_grid(modes)[None, :]

    def fourier_coefficients(q):
        # A periodic trapezoid rule is spectrally accurate away from q=1 and
        # remains well behaved for the mild integrable cusp as q approaches 1.
        pair_factor = ((1.0 - q) ** 2 + 4.0 * q * sin_squared) ** alpha
        return np.fft.rfft(pair_factor, axis=1) / modes

    products = (fourier_coefficients(q1) * fourier_coefficients(q2)
                * fourier_coefficients(q3))
    return (products[:, 0] + 2.0 * np.sum(products[:, 1:-1], axis=1)
            + products[:, -1]).real


def three_ion_coefficient(gamma, ratio, order=40):
    """Three-ion coefficient using ordered log-radial Gauss--Legendre nodes."""
    length = np.log(ratio)
    x, w = nodes(order)
    first_gap = np.broadcast_to((length * x)[:, None], (order, order)).ravel()
    second_gap = ((length - length * x)[:, None] * x[None, :]).ravel()
    weights = (length * (length - length * x)[:, None] * w[:, None] * w[None, :]).ravel()
    span = length - first_gap - second_gap
    angular = _three_angle_average(gamma, first_gap, second_gap)

    common_slope = 6.0 - 6.0 * gamma
    radial_weight = weights * angular * np.exp(
        (4.0 - 3.0 * gamma) * first_gap + (2.0 - gamma) * second_gap
    )
    partition = np.dot(radial_weight, span * exprel(common_slope * span))
    boundary = np.dot(radial_weight, np.exp(common_slope * span))
    return -np.expm1(-2.0 * length) * boundary / (6.0 * partition)
