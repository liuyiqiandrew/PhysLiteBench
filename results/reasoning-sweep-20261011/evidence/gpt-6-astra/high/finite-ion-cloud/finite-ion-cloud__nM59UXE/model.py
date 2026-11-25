"""Canonical finite-annulus osmotic coefficients for one to three ions.

With L = log(R/a), the wall pressure gives
    phi = (1 - exp(-2*L)) * d(log Z)/dL / (2*N).
All lengths in Z can be scaled by a, so the disk radius cancels from phi.
"""

from functools import lru_cache

import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize_scalar
from scipy.special import exprel, hyp2f1, roots_legendre


@lru_cache(maxsize=4096)
def coefficient(gamma, count, ratio):
    """Evaluate the finite-cell response at the actual coupling Gamma."""
    if count == 3:
        return three_ion_coefficient(gamma, ratio)
    length = np.log(ratio)
    area_factor = -np.expm1(-2 * length)
    if count == 1:
        slope = 2 - gamma
        return area_factor / (2 * length * exprel(-slope * length))
    if count != 2:
        raise ValueError("ion_count must be 1, 2, or 3")

    # Order the log radii as t, t+d, then integrate over t analytically.
    slope, gap_slope = 4 - 3 * gamma, 2 - gamma

    def angular(gap):
        return hyp2f1(-gamma / 2, -gamma / 2, 1, np.exp(-2 * gap))

    partition = quad(
        lambda d: angular(d) * np.exp(gap_slope * d)
        * (length - d) * exprel(slope * (length - d)),
        0, length, epsabs=1e-11, epsrel=1e-11,
    )[0]
    boundary = quad(
        lambda d: angular(d) * np.exp(gap_slope * d + slope * (length - d)),
        0, length, epsabs=1e-11, epsrel=1e-11,
    )[0]
    return area_factor * boundary / (4 * partition)


def predict_at(experiments, coupling):
    return np.asarray([
        coefficient(e['temperature_multiplier'] * coupling,
                    e['ion_count'], e['radius_ratio'])
        for e in experiments
    ], dtype=float)


class Model:
    def __init__(self):
        self.coupling = None

    def fit(self, records):
        """Fit the shared g by Gaussian maximum likelihood on [0.7, 1.4]."""
        records = list(records)
        if not records:
            raise ValueError("at least one calibration record is required")
        experiments = [r['input'] for r in records]
        values = np.asarray([r['value'] for r in records], dtype=float)
        sigma = np.asarray([r['sigma'] for r in records], dtype=float)
        if not np.all(np.isfinite(values)):
            raise ValueError("calibration values must be finite")
        if not np.all(np.isfinite(sigma) & (sigma > 0)):
            raise ValueError("calibration sigma must be finite and positive")

        def loss(coupling):
            residual = (predict_at(experiments, coupling) - values) / sigma
            return np.dot(residual, residual)

        result = minimize_scalar(loss, bounds=(0.7, 1.4), method='bounded',
                                 options={'xatol': 1e-11})
        if not result.success:
            raise RuntimeError("coupling fit did not converge")
        # A constrained optimum can also lie exactly on either endpoint.
        self.coupling = float(min((0.7, result.x, 1.4), key=loss))
        return self

    def predict(self, experiments):
        if self.coupling is None:
            raise ValueError("fit the model before predicting")
        return predict_at(experiments, self.coupling)


@lru_cache(maxsize=16)
def nodes(order):
    x, w = roots_legendre(order)
    return (x + 1) / 2, w / 2


def _three_ion_angular(gamma, d1, d2, modes=48):
    """Joint angular average, retaining the closed three-pair correlation.

    For f(q,theta) = |1-q*exp(i*theta)|**Gamma, its Fourier coefficient is
      c_k(q) = (-s)_k/k! * q**k * 2F1(-s,k-s;k+1;q*q), s=Gamma/2.
    The three relative angles are dependent. Their joint integral is
      c_0(q1)c_0(q2)c_0(q1*q2) + 2 sum_{k>=1} c_k(q1)c_k(q2)c_k(q1*q2).
    The product series converges even at coincident radii; 48 modes give
    sub-micro precision throughout the stated domain. At Gamma=2 or 4
    the series terminates exactly.
    """
    s = gamma / 2
    k = np.arange(modes + 1)[None, :]
    prefactor = np.ones(modes + 1)
    for j in range(modes):
        prefactor[j + 1] = prefactor[j] * (j - s) / (j + 1)

    product = np.ones((len(d1), modes + 1))
    for gap in (d1, d2, d1 + d2):
        q = np.exp(-gap[:, None])
        product *= (prefactor * q**k
                    * hyp2f1(-s, k - s, k + 1, q*q))
    return product[:, 0] + 2 * np.sum(product[:, 1:], axis=1)


def three_ion_coefficient(gamma, ratio, order=48):
    """Integrate ordered log radii t, t+d1, t+d1+d2 over the finite cell."""
    length = np.log(ratio)
    x, w = nodes(order)
    d1 = np.broadcast_to((length * x)[:, None], (order, order)).ravel()
    d2 = ((length - length * x)[:, None] * x[None, :]).ravel()
    weights = (length * (length - length * x)[:, None]
               * w[:, None] * w[None, :]).ravel()
    angular = _three_ion_angular(gamma, d1, d2)

    # The common translation t has exponent 6-6*Gamma. Its integral is
    # span*exprel(slope*span), including slope=0 without cancellation.
    # Differentiating this integral in L gives the outer-wall numerator;
    # the moving triangle edge contributes zero since span=0 there.
    span = length - d1 - d2
    slope = 6 - 6 * gamma
    weight = weights * angular * np.exp((4 - 3 * gamma) * d1
                                       + (2 - gamma) * d2)
    partition = np.dot(weight, span * exprel(slope * span))
    boundary = np.dot(weight, np.exp(slope * span))
    return -np.expm1(-2 * length) * boundary / (6 * partition)
