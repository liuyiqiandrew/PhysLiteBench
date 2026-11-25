"""Canonical finite-cell pressure for up to three logarithmically interacting ions.

Writing L = log(R/a), the pressure is
    phi = (1 - exp(-2L)) / (2N) * d(log Z)/dL.
The factor a**(2N) in Z drops out of this derivative, so the answer depends
on the radius ratio, not the absolute disk radius. Ordered log radii allow
one radial integral (a common translation) to be performed analytically.
"""

from functools import lru_cache

import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize_scalar
from scipy.special import exprel, hyp2f1, roots_legendre


@lru_cache(maxsize=4096)
def coefficient(gamma, count, ratio):
    """Return the finite-annulus osmotic coefficient at coupling gamma."""
    if count == 3:
        return three_ion_coefficient(gamma, ratio)
    length = np.log(ratio)
    area_factor = -np.expm1(-2 * length)
    if count == 1:
        slope = 2 - gamma
        return area_factor * np.exp(slope * length) / (
            2 * length * exprel(slope * length)
        )
    if count != 2:
        raise ValueError("ion_count must be 1, 2, or 3")

    slope, gap_slope = 4 - 3 * gamma, 2 - gamma

    def angular_weight(gap):
        return hyp2f1(-gamma / 2, -gamma / 2, 1, np.exp(-2 * gap))

    partition = quad(
        lambda gap: angular_weight(gap) * np.exp(gap_slope * gap)
        * (length - gap) * exprel(slope * (length - gap)),
        0, length, epsabs=1e-11, epsrel=1e-11,
    )[0]
    boundary = quad(
        lambda gap: angular_weight(gap)
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
        """Fit the shared g by Gaussian maximum likelihood on [0.7, 1.4]."""
        records = list(records)
        if not records:
            raise ValueError("at least one calibration record is required")
        experiments = [r["input"] for r in records]
        values = np.asarray([r["value"] for r in records], dtype=float)
        sigmas = np.asarray([r["sigma"] for r in records], dtype=float)
        if not np.isfinite(values).all() or not np.isfinite(sigmas).all() or np.any(sigmas <= 0):
            raise ValueError("calibration values must be finite and sigmas positive")

        def loss(coupling):
            residual = (predict_at(experiments, coupling) - values) / sigmas
            return np.dot(residual, residual)

        result = minimize_scalar(loss, bounds=(0.7, 1.4), method="bounded",
                                 options={"xatol": 1e-10})
        if not result.success:
            raise RuntimeError("coupling fit failed: " + result.message)
        # A constrained maximum-likelihood estimate can lie at either endpoint.
        self.coupling = float(min((result.x, 0.7, 1.4), key=loss))
        return self

    def predict(self, experiments):
        if self.coupling is None:
            raise ValueError("fit the model before predicting")
        return predict_at(experiments, self.coupling)


@lru_cache(maxsize=16)
def nodes(order):
    x, w = roots_legendre(order)
    return (x + 1) / 2, w / 2


def _three_ion_angular(gamma, d1, d2, modes=64):
    """Joint angular average, including the closed triangle of pair factors.

    For f(t, theta) = |1 - t exp(i theta)|**gamma, its Fourier
    coefficient is c_k(t) = (-p)_k t**k / k! *
    2F1(k-p, -p; k+1; t**2), with p = gamma/2 and c_-k = c_k.
    Integrating the two independent angles gives
        c_0(t1)c_0(t2)c_0(t1*t2) + 2 sum_{k>=1} c_k(t1)c_k(t2)c_k(t1*t2).
    Multiplying three separate pair averages would omit this sum.
    """
    gaps = np.stack((d1, d2, d1 + d2))
    squared_ratios = np.exp(-2 * gaps)
    p = gamma / 2
    angular = np.prod(hyp2f1(-p, -p, 1, squared_ratios), axis=0)
    prefactor = 1.0
    for k in range(1, modes + 1):
        prefactor *= (k - 1 - p) / k
        if prefactor == 0:
            break  # At even integer gamma the Fourier series terminates.
        fourier = (prefactor * np.exp(-k * gaps)
                   * hyp2f1(k - p, -p, k + 1, squared_ratios))
        angular += 2 * np.prod(fourier, axis=0)
    return angular


def three_ion_coefficient(gamma, ratio, order=48):
    """Integrate ordered radial gaps after the full joint angular average."""
    length = np.log(ratio)
    x, w = nodes(order)
    d1 = np.broadcast_to((length * x)[:, None], (order, order)).ravel()
    d2 = ((length * (1 - x))[:, None] * x[None, :]).ravel()
    weights = (length**2 * (1 - x)[:, None] * w[:, None] * w[None, :]).ravel()
    angular = _three_ion_angular(gamma, d1, d2)
    span = length - d1 - d2
    # Log radii are s, s+d1, s+d1+d2, with 0 < s < span.
    slope = 6 - 6 * gamma
    weight = weights * angular * np.exp((4 - 3 * gamma) * d1 + (2 - gamma) * d2)
    partition = np.dot(weight, span * exprel(slope * span))
    # Differentiation of the common-translation integral gives the wall contact.
    # Its vanishing value at span=0 removes the moving-gap-domain boundary term.
    boundary = np.dot(weight, np.exp(slope * span))
    return -np.expm1(-2 * length) * boundary / (6 * partition)
