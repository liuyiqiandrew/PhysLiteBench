"""Canonical osmotic pressure of one, two, or three ions in a finite cell.

With L = log(R/a), the pressure is
    phi = (1 - exp(-2*L)) / (2*N) * d(log Z)/dL.
Scaling positions by a removes the disk radius from this expression.
The radial integrals below use ordered log radii and integrate their
common translation analytically, including its outer-wall derivative.
"""

from functools import lru_cache

import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize_scalar
from scipy.special import exprel, hyp2f1, roots_legendre


@lru_cache(maxsize=4096)
def coefficient(gamma, count, ratio):
    """Return phi at the actual coupling Gamma = u*g and finite R/a."""
    if count == 3:
        return three_ion_coefficient(gamma, ratio)
    if count not in (1, 2):
        raise ValueError("ion_count must be 1, 2, or 3")

    length = np.log(ratio)
    area_factor = -np.expm1(-2 * length)
    if count == 1:
        slope = 2 - gamma
        # exprel also supplies the regular limit at Gamma = 2.
        return area_factor / (2 * length * exprel(-slope * length))

    translation_slope, gap_slope = 4 - 3 * gamma, 2 - gamma

    def angular(gap):
        return hyp2f1(-gamma / 2, -gamma / 2, 1, np.exp(-2 * gap))

    partition = quad(
        lambda gap: angular(gap)
        * np.exp(gap_slope * gap)
        * (length - gap)
        * exprel(translation_slope * (length - gap)),
        0, length, epsabs=1e-11, epsrel=1e-11,
    )[0]
    boundary = quad(
        lambda gap: angular(gap)
        * np.exp(gap_slope * gap + translation_slope * (length - gap)),
        0, length, epsabs=1e-11, epsrel=1e-11,
    )[0]
    return area_factor * boundary / (4 * partition)


def predict_at(experiments, coupling):
    return np.asarray([
        coefficient(
            e["temperature_multiplier"] * coupling,
            e["ion_count"],
            e["radius_ratio"],
        )
        for e in experiments
    ], dtype=float)


class Model:
    def __init__(self):
        self.coupling = None

    def fit(self, records):
        """Fit the shared reference coupling by Gaussian maximum likelihood."""
        records = list(records)
        if not records:
            raise ValueError("at least one calibration record is required")
        experiments = [record["input"] for record in records]
        values = np.asarray([record["value"] for record in records], dtype=float)
        sigmas = np.asarray([record["sigma"] for record in records], dtype=float)
        if not np.all(np.isfinite(values)):
            raise ValueError("calibration values must be finite")
        if not np.all(np.isfinite(sigmas) & (sigmas > 0)):
            raise ValueError("calibration sigmas must be finite and positive")

        def objective(coupling):
            residuals = (predict_at(experiments, coupling) - values) / sigmas
            return np.dot(residuals, residuals)

        result = minimize_scalar(
            objective, bounds=(0.7, 1.4), method="bounded",
            options={"xatol": 1e-11},
        )
        if not result.success:
            raise RuntimeError("coupling fit failed: " + result.message)
        # A bounded optimizer searches the interior; endpoints are also allowed.
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


def _three_ion_angular(gamma, d1, d2, modes=32):
    """Joint angular average of all three pair Boltzmann factors.

    For f_q(theta) = |1 - q*exp(i*theta)|**Gamma, its Fourier
    coefficient is
        c_m(q) = q**m * (-Gamma/2)_m / m!
                 * 2F1(-Gamma/2, m-Gamma/2; m+1; q**2).
    Integrating over the two independent angles leaves
        c_0(q1)*c_0(q2)*c_0(q1*q2)
        + 2*sum_{m>=1} c_m(q1)*c_m(q2)*c_m(q1*q2).
    The nonzero modes retain the three-body angular correlations.
    Their product decays at least as m**(-3*Gamma-3); 32 modes
    suffice over the stated domain, even when both radial gaps vanish.
    """
    gaps = np.array(np.broadcast_arrays(d1, d2, d1 + d2))
    q = np.exp(-gaps)
    z = q * q
    alpha = gamma / 2
    angular = np.prod(hyp2f1(-alpha, -alpha, 1, z), axis=0)
    prefactor = np.ones_like(q)
    for m in range(1, modes + 1):
        prefactor *= q * ((m - 1 - alpha) / m)
        if m - 1 == alpha:
            # At even integer Gamma the Fourier series terminates exactly.
            break
        harmonic = prefactor * hyp2f1(-alpha, m - alpha, m + 1, z)
        angular += 2 * np.prod(harmonic, axis=0)
    return angular


def three_ion_coefficient(gamma, ratio, order=32):
    """Integrate the correlated three-ion ensemble over ordered log radii."""
    length = np.log(ratio)
    x, w = nodes(order)
    # log(r_i/a) = (t, t+d1, t+d1+d2), with 0 < t < L-d1-d2.
    d1 = np.broadcast_to((length * x)[:, None], (order, order)).ravel()
    d2 = ((length - length * x)[:, None] * x[None, :]).ravel()
    weights = (
        length * (length - length * x)[:, None] * w[:, None] * w[None, :]
    ).ravel()
    angular = _three_ion_angular(gamma, d1, d2)
    span = length - d1 - d2
    slope = 6 - 6 * gamma
    weight = weights * angular * np.exp((4 - 3 * gamma) * d1 + (2 - gamma) * d2)
    partition = np.dot(weight, span * exprel(slope * span))
    boundary = np.dot(weight, np.exp(slope * span))
    return -np.expm1(-2 * length) * boundary / (6 * partition)
