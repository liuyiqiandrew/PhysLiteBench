import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import hyp2f1, roots_legendre


# Fixed Gauss-Legendre nodes are sufficient here because the transformed
# two-ion integral is smooth on the unit square. Keeping them at module scope
# also makes prediction of a batch of experiments inexpensive.
_GL_N = 64
_GL_X, _GL_W = roots_legendre(_GL_N)
_GL_X = (_GL_X + 1.0) / 2.0
_GL_W = _GL_W / 2.0


def _one_ion_coefficient(gamma, ratio):
    """Finite-annulus osmotic coefficient for one ion."""
    length = np.log(ratio)
    slope = 2.0 - gamma

    # The expression is
    # (1-exp(-2l)) (2-gamma) / (2(1-exp(-(2-gamma)l))).
    # The explicit limit avoids cancellation at gamma=2.
    numerator = -np.expm1(-2.0 * length)
    if abs(slope * length) < 1.0e-8:
        return numerator / (2.0 * length)
    denominator = -np.expm1(-slope * length)
    return numerator * slope / (2.0 * denominator)


def _two_ion_coefficient(gamma, ratio):
    """Finite-annulus osmotic coefficient for two ions.

    After integrating over the common angular rotation and ordering the two
    radii x >= y, the angular factor is

        average_theta |x exp(i theta) - y|**gamma
        = x**gamma * 2F1(-gamma/2, -gamma/2; 1; (y/x)**2).

    The remaining triangular radial integral is mapped to a unit square for
    stable fixed-order quadrature.
    """
    length = ratio - 1.0
    u = _GL_X[:, None]
    v = _GL_X[None, :]
    x = 1.0 + length * u
    y = 1.0 + length * u * v

    angular = hyp2f1(-gamma / 2.0, -gamma / 2.0, 1.0, (y / x) ** 2)
    integrand = x ** (1.0 - gamma) * y ** (1.0 - 2.0 * gamma) * angular
    weights = _GL_W[:, None] * _GL_W[None, :]
    partition = length**2 * np.sum(weights * u * integrand)

    # This is the derivative of the ordered radial partition integral with
    # respect to its upper radius.
    boundary_x = 1.0 + length * _GL_X
    boundary_angular = hyp2f1(
        -gamma / 2.0,
        -gamma / 2.0,
        1.0,
        (boundary_x / ratio) ** 2,
    )
    boundary = length * np.sum(
        _GL_W
        * boundary_x ** (1.0 - 2.0 * gamma)
        * boundary_angular
    )
    derivative = ratio ** (1.0 - gamma) * boundary

    # phi = (A/(2 N pi R)) d(log Z)/dR, with N=2 and R/a=ratio.
    return (ratio**2 - 1.0) * derivative / (4.0 * ratio * partition)


def coefficient(gamma, count, ratio):
    """Return the canonical finite-cell osmotic coefficient."""
    if count == 1:
        return _one_ion_coefficient(float(gamma), float(ratio))
    if count == 2:
        return _two_ion_coefficient(float(gamma), float(ratio))
    raise ValueError("ion_count must be 1 or 2")


def predict_at(experiments, coupling):
    return np.asarray(
        [
            coefficient(
                e["temperature_multiplier"] * coupling,
                e["ion_count"],
                e["radius_ratio"],
            )
            for e in experiments
        ],
        dtype=float,
    )


class Model:
    def __init__(self):
        self.coupling = None

    def fit(self, records):
        if not records:
            raise ValueError("at least one calibration record is required")

        experiments = [record["input"] for record in records]
        values = np.asarray([record["value"] for record in records], dtype=float)
        sigmas = np.asarray([record["sigma"] for record in records], dtype=float)
        if np.any(~np.isfinite(values)) or np.any(~np.isfinite(sigmas)) or np.any(sigmas <= 0.0):
            raise ValueError("calibration values and sigmas must be finite and sigmas positive")

        def objective(coupling):
            residual = (predict_at(experiments, coupling) - values) / sigmas
            return float(np.dot(residual, residual))

        result = minimize_scalar(
            objective,
            bounds=(0.7, 1.4),
            method="bounded",
            options={"xatol": 1.0e-12},
        )
        # Include the endpoints explicitly: a constrained optimum can occur
        # at either boundary, and bounded minimization does not guarantee that
        # they are sampled.
        candidates = [(result.fun, result.x), (objective(0.7), 0.7), (objective(1.4), 1.4)]
        self.coupling = float(min(candidates, key=lambda item: item[0])[1])
        return self

    def predict(self, experiments):
        return predict_at(experiments, self.coupling)
