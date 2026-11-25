"""Linear diffusiophoretic response of a sphere with Navier slip.

The concentration problem is independent of the flow in the dilute,
rapid-diffusion limit.  ``response`` returns the mobility multiplied by the
viscosity; ``predict_at`` supplies the remaining 1 / viscosity factor.
"""

from functools import lru_cache

import numpy as np
from scipy.integrate import quad, solve_bvp


@lru_cache(maxsize=512)
def response(radius, width, strength, slip):
    """Return viscosity times the velocity response for one preparation."""
    radius = float(radius)
    width = float(width)
    strength = float(strength)
    slip = float(slip)
    if strength == 0.0:
        return 0.0

    outer = radius + width

    # Write c_1=exp(-U) f(r) cos(theta) and x=(r-a)/w.
    def equations(x, y):
        r = radius + width * x
        # f'' + (2/r - U') f' - 2 f/r^2 = 0, U'=-2s(1-x)/w.
        return np.vstack((
            y[1],
            -(2.0 * width / r + 2.0 * strength * (1.0 - x)) * y[1]
            + 2.0 * width**2 * y[0] / r**2,
        ))

    def boundary(left, right):
        # f'(a)=0 and f(r)=r+A/r^2 outside the potential range.
        return np.array((left[1], outer * right[1] / width + 2.0 * right[0] - 3.0 * outer))

    grid = np.linspace(0.0, 1.0, 129)
    r = radius + width * grid
    initial = np.array((r + radius**3 / (2.0 * r**2), width * (1.0 - radius**3 / r**3)))
    solution = solve_bvp(equations, boundary, grid, initial, tol=2e-10, max_nodes=10000)
    if not solution.success:
        raise RuntimeError("solute boundary-value problem failed: " + solution.message)

    # Auxiliary translating Navier-slip sphere:
    # uhat_r/cos(theta)=alpha*a/r-gamma*a^3/r^3,
    # gamma=1/[2(1+3b/a)], alpha=1+gamma,
    # D/(6*pi*mu*a)=(1+2b/a)/(1+3b/a).
    beta = slip / radius
    gamma = 1.0 / (2.0 * (1.0 + 3.0 * beta))
    alpha = 1.0 + gamma
    drag_ratio = (1.0 + 2.0 * beta) / (1.0 + 3.0 * beta)

    def integrand(r):
        x = (r - radius) / width
        concentration_factor = np.exp(-strength * (1.0 - x)**2) * solution.sol(x)[0]
        potential_derivative = -2.0 * strength * (1.0 - x) / width
        # r^2 is the direct reaction from the potential carried by the
        # particle; the rest is the auxiliary Navier-slip flow contribution.
        reciprocal_kernel = r**2 - alpha * radius * r + gamma * radius**3 / r
        return reciprocal_kernel * concentration_factor * potential_derivative

    integral, _ = quad(integrand, radius, outer, epsabs=2e-11, epsrel=2e-11, limit=200)
    return 2.0 * integral / (9.0 * radius * drag_ratio)


def predict_at(experiments, viscosity):
    """Predict responses for an iterable of documented input mappings."""
    if viscosity is None or not np.isfinite(viscosity) or viscosity <= 0.0:
        raise ValueError("viscosity must be a positive finite number")
    return np.asarray(
        [response(e["radius"], e["width"], e["strength"], e["slip"]) for e in experiments],
        dtype=float,
    ) / float(viscosity)


class Model:
    def __init__(self):
        self.viscosity = None

    def fit(self, records):
        """Infer viscosity by weighted least squares from calibration records."""
        records = list(records)
        if not records:
            raise ValueError("at least one calibration record is required")

        base_response = np.asarray(
            [response(r["input"]["radius"], r["input"]["width"],
                      r["input"]["strength"], r["input"]["slip"])
             for r in records],
            dtype=float,
        )
        observed = np.asarray([r["value"] for r in records], dtype=float)
        uncertainty = np.asarray([r["sigma"] for r in records], dtype=float)
        if (not np.isfinite(base_response).all() or not np.isfinite(observed).all()
                or not np.isfinite(uncertainty).all() or np.any(uncertainty <= 0.0)):
            raise ValueError("calibration records must contain finite values and positive sigma")

        # y_i=q_i/mu is linear in inverse viscosity.
        weights = uncertainty ** -2
        denominator = np.sum(weights * base_response**2)
        if denominator == 0.0:
            raise ValueError("calibration records do not identify viscosity")
        inverse_viscosity = np.sum(weights * base_response * observed) / denominator
        if not np.isfinite(inverse_viscosity) or inverse_viscosity <= 0.0:
            raise ValueError("calibration data imply a non-positive viscosity")

        self.viscosity = float(np.clip(1.0 / inverse_viscosity, 0.8, 1.6))
        return self

    def predict(self, experiments):
        return predict_at(experiments, self.viscosity)
