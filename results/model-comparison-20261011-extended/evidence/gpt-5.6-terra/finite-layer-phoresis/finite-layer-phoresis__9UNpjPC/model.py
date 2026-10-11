"""Linear diffusiophoretic response of a sphere with Navier slip.

The concentration problem is independent of viscosity in the rapid-diffusion
limit.  Thus each response is ``mobility / viscosity``, and calibration is a
weighted linear fit in inverse viscosity.
"""

from functools import lru_cache

import numpy as np
from scipy.integrate import quad, solve_bvp


def _as_finite_float(value, name):
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite scalar") from exc
    if not np.isfinite(value):
        raise ValueError(f"{name} must be a finite scalar")
    return value


def _parameters(experiment):
    """Read and validate a documented experiment input."""
    try:
        radius = _as_finite_float(experiment["radius"], "radius")
        width = _as_finite_float(experiment["width"], "width")
        strength = _as_finite_float(experiment["strength"], "strength")
        slip = _as_finite_float(experiment["slip"], "slip")
    except (KeyError, TypeError) as exc:
        raise ValueError(
            "each experiment must contain radius, width, strength, and slip"
        ) from exc
    if radius <= 0.0 or width <= 0.0 or slip < 0.0:
        raise ValueError("radius and width must be positive and slip non-negative")
    return radius, width, strength, slip


@lru_cache(maxsize=512)
def _mobility(radius, width, strength, slip):
    """Return response times viscosity for one physical parameter set."""
    if strength == 0.0:
        return 0.0

    outer_radius = radius + width

    def equations(x, y):
        r = radius + width * x
        return np.array(
            [
                y[1],
                -(2.0 * width / r + 2.0 * strength * (1.0 - x)) * y[1]
                + 2.0 * width**2 * y[0] / r**2,
            ]
        )

    def boundary(left, right):
        # No normal solute flux and matching to the outer r + A/r^2 field.
        return np.array(
            [left[1], outer_radius * right[1] / width + 2.0 * right[0] - 3.0 * outer_radius]
        )

    grid = np.linspace(0.0, 1.0, 129)
    r_grid = radius + width * grid
    initial = np.array(
        [r_grid + radius**3 / (2.0 * r_grid**2), width * (1.0 - radius**3 / r_grid**3)]
    )
    solution = solve_bvp(equations, boundary, grid, initial, tol=2e-10, max_nodes=10000)
    if not solution.success:
        raise RuntimeError(f"solute boundary-value solve failed: {solution.message}")

    beta = slip / radius
    drag_ratio = (1.0 + 2.0 * beta) / (1.0 + 3.0 * beta)
    # Radial velocity of the reciprocal translating Navier-slip sphere is
    # A/r+B/r^3.  r^2(1-u_r/U) is the force-density kernel.
    a_coefficient = 1.5 * radius * (1.0 + 2.0 * beta) / (1.0 + 3.0 * beta)
    b_coefficient = -0.5 * radius**3 / (1.0 + 3.0 * beta)

    def integrand(r):
        x = (r - radius) / width
        concentration = np.exp(-strength * (1.0 - x) ** 2) * solution.sol(x)[0]
        potential_derivative = -2.0 * strength * (1.0 - x) / width
        reciprocal_kernel = r**2 - a_coefficient * r - b_coefficient / r
        return reciprocal_kernel * concentration * potential_derivative

    integral, _ = quad(
        integrand, radius, outer_radius, epsabs=2e-11, epsrel=2e-11, limit=100
    )
    return 2.0 * integral / (9.0 * radius * drag_ratio)


def response(radius, width, strength, slip):
    """Compatibility helper returning the response at unit viscosity."""
    return _mobility(*_parameters({
        "radius": radius, "width": width, "strength": strength, "slip": slip
    }))


def predict_at(experiments, viscosity):
    """Predict responses for experiments at a specified positive viscosity."""
    viscosity = _as_finite_float(viscosity, "viscosity")
    if viscosity <= 0.0:
        raise ValueError("viscosity must be positive")
    return np.asarray(
        [_mobility(*_parameters(experiment)) / viscosity for experiment in experiments],
        dtype=float,
    )


class Model:
    def __init__(self):
        self.viscosity = None

    def fit(self, records):
        """Maximize the stated independent-Gaussian likelihood."""
        mobilities, observations, weights = [], [], []
        for record in records:
            try:
                experiment = record["input"]
                value = _as_finite_float(record["value"], "record value")
                sigma = _as_finite_float(record["sigma"], "record sigma")
            except (KeyError, TypeError) as exc:
                raise ValueError("each record must contain input, value, and sigma") from exc
            if sigma <= 0.0:
                raise ValueError("record sigma must be positive")
            mobilities.append(_mobility(*_parameters(experiment)))
            observations.append(value)
            weights.append(1.0 / sigma**2)

        if not mobilities:
            raise ValueError("at least one calibration record is required")
        mobilities = np.asarray(mobilities, dtype=float)
        observations = np.asarray(observations, dtype=float)
        weights = np.asarray(weights, dtype=float)
        denominator = np.sum(weights * mobilities**2)
        if not np.isfinite(denominator) or denominator <= 0.0:
            raise ValueError("calibration records contain no viscosity information")

        # Fit q=1/viscosity, where value = mobility*q, respecting its known range.
        inverse_viscosity = np.sum(weights * mobilities * observations) / denominator
        inverse_viscosity = float(np.clip(inverse_viscosity, 1.0 / 1.6, 1.0 / 0.8))
        self.viscosity = 1.0 / inverse_viscosity
        return self

    def predict(self, experiments):
        if self.viscosity is None:
            raise RuntimeError("fit must be called before predict")
        return predict_at(experiments, self.viscosity)
