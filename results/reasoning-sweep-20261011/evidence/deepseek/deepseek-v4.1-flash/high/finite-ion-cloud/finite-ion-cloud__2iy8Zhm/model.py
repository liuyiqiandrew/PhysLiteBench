import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize_scalar
from scipy.special import exprel, hyp2f1, roots_legendre
from functools import lru_cache

# ---------------------------------------------------------------------------
# Finite-cell osmotic coefficient
#
# A rigid disk of radius ``a`` carries charge ``+N*q`` while ``N`` mobile
# point ions of charge ``-q`` (N = 1, 2, 3) move in the annulus a < r < R.
# The reduced configurational energy of a configuration is
#
#     U/(k_B T) = Gamma * [ N*sum_i log(r_i/a) - sum_{i<j} log(|r_i-r_j|/a) ],
#     Gamma = u * g,
#
# with u the temperature multiplier and g the unknown reference coupling.
# The outer pressure P = -dF/dA (A = pi*(R^2-a^2)) defines the dimensionless
# osmotic coefficient phi = P/(k_B T nbar), nbar = N/A.  Because the energy
# and the domain are scale invariant in ``a``, phi depends only on
# (Gamma, N, R/a); the ``disk_radius`` therefore does not enter the response.
#
# The closed forms / quadratures below evaluate that phi in the canonical
# ensemble at finite R/a (no dilution or large-N limit).
# ---------------------------------------------------------------------------


@lru_cache(maxsize=None)
def coefficient(gamma, count, ratio):
    if count == 3:
        return three_ion_coefficient(gamma, ratio)
    length = np.log(ratio)
    if count == 1:
        slope = 2 - gamma
        return -np.expm1(-2 * length) * np.exp(slope * length) / (
            2 * length * exprel(slope * length)
        )
    a, b = 4 - 3 * gamma, 2 - gamma

    def angular(gap):
        return hyp2f1(-gamma / 2, -gamma / 2, 1, np.exp(-2 * gap))

    partition = quad(
        lambda d: angular(d) * np.exp(b * d) * (length - d) * exprel(a * (length - d)),
        0, length, epsabs=1e-11, epsrel=1e-11,
    )[0]
    boundary = quad(
        lambda d: angular(d) * np.exp(b * d + a * (length - d)),
        0, length, epsabs=1e-11, epsrel=1e-11,
    )[0]
    return -np.expm1(-2 * length) * boundary / (4 * partition)


def predict_at(experiments, coupling):
    return np.asarray(
        [
            coefficient(e["temperature_multiplier"] * coupling,
                        e["ion_count"], e["radius_ratio"])
            for e in experiments
        ],
        dtype=float,
    )


class Model:
    """Calibrated finite-cell osmotic-coefficient model."""

    # Physical bracket for the shared reference coupling g.
    COUPLING_BOUNDS = (0.7, 1.4)

    def __init__(self):
        self.coupling = None

    # -- calibration -------------------------------------------------------
    def fit(self, records):
        """Infer the shared electrostatic coupling g from noisy records.

        Each record supplies ``input`` (with ``temperature_multiplier``,
        ``ion_count``, ``radius_ratio``, ``disk_radius``), the measured
        ``value`` of phi and its additive Gaussian ``sigma``.  The mean
        chi-square residual is minimised over the physical bracket.
        """
        lo, hi = self.COUPLING_BOUNDS
        inputs = [r["input"] for r in records]
        values = np.asarray([r["value"] for r in records], dtype=float)
        sigma = np.asarray(
            [r.get("sigma", 1.0) for r in records], dtype=float
        )
        sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, 1.0)

        def chi_square(g):
            residual = (predict_at(inputs, g) - values) / sigma
            return float(np.mean(residual ** 2))

        result = minimize_scalar(
            chi_square, bounds=(lo, hi), method="bounded",
            options={"xatol": 1e-10, "maxiter": 500},
        )
        self.coupling = float(np.clip(result.x, lo, hi))
        return self

    # -- prediction --------------------------------------------------------
    def predict(self, experiments):
        if self.coupling is None:
            raise RuntimeError("Model.fit must be called before predict")
        experiments = list(experiments)
        if not experiments:
            return np.empty(0, dtype=float)
        return predict_at(experiments, self.coupling)


# ---------------------------------------------------------------------------
# Three-ion response (2-D Gauss-Legendre quadrature over pair separations).
# ---------------------------------------------------------------------------


@lru_cache(maxsize=None)
def nodes(order):
    x, w = roots_legendre(order)
    return (x + 1) / 2, w / 2


def three_ion_coefficient(gamma, ratio, order=32):
    length = np.log(ratio)
    x, w = nodes(order)
    d1 = np.broadcast_to((length * x)[:, None], (order, order)).ravel()
    d2 = ((length - length * x)[:, None] * x[None, :]).ravel()
    weights = (length * (length - length * x)[:, None] * w[:, None] * w[None, :]).ravel()
    angular = np.ones(len(d1))
    for gap in [d1, d2, d1 + d2]:
        angular *= hyp2f1(-gamma / 2, -gamma / 2, 1, np.exp(-2 * gap))
    span = length - d1 - d2
    slope = 6 - 6 * gamma
    weight = weights * angular * np.exp((4 - 3 * gamma) * d1 + (2 - gamma) * d2)
    partition = np.dot(weight, span * exprel(slope * span))
    boundary = np.dot(weight, np.exp(slope * span))
    return -np.expm1(-2 * length) * boundary / (6 * partition)
