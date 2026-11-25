"""Linear diffusiophoretic response of a sphere with a finite interaction layer.

The concentration problem is independent of the viscosity in the stated
rapid-diffusion limit.  ``response`` therefore returns the response with
viscosity one; the factor of viscosity is applied only in ``predict_at``.
"""

from functools import lru_cache

import numpy as np
from scipy.integrate import quad
from scipy.integrate import solve_bvp


@lru_cache(maxsize=256)
def response(radius, width, strength, slip):
    """Return the velocity/gradient response at unit dynamic viscosity.

    Write the angular part of the concentration as
    ``exp(-U) * f(r) * cos(theta)``.  The steady zero-flux solute equation
    gives a scalar boundary-value problem for ``f`` in the interaction
    layer.  Outside that layer its solution is harmonic, which supplies the
    matching condition at ``radius + width``.

    The final integral follows from the reciprocal theorem.  In particular,
    the test flow is that of a translating Navier-slip sphere.  Its radial
    velocity is ``q*a/r + (1-q)*a**3/r**3``, where
    ``q = 3(1+2b/a)/(2(1+3b/a))``.  Retaining this dependence in the kernel
    matters for finite slip; reducing it to a drag correction alone is only
    valid at zero slip.
    """
    radius = float(radius)
    width = float(width)
    strength = float(strength)
    slip = float(slip)

    if strength == 0.0:
        return 0.0

    layer_edge = radius + width

    def equations(x, y):
        r = radius + width * x
        # d/dr[exp(-U) grad(f cos(theta))] = 0, in x coordinates.
        return np.array(
            [
                y[1],
                -(2.0 * width / r + 2.0 * strength * (1.0 - x)) * y[1]
                + 2.0 * width**2 * y[0] / r**2,
            ]
        )

    def boundary(left, right):
        # No normal solute flux at the sphere, followed by matching to
        # f = r + A/r**2 beyond the interaction layer.
        return np.array(
            [left[1], layer_edge * right[1] / width + 2.0 * right[0] - 3.0 * layer_edge]
        )

    grid = np.linspace(0.0, 1.0, 129)
    r_grid = radius + width * grid
    # The zero-potential solution is an accurate, smooth starting solution.
    initial = np.array(
        [r_grid + radius**3 / (2.0 * r_grid**2), width * (1.0 - radius**3 / r_grid**3)]
    )
    solution = solve_bvp(equations, boundary, grid, initial, tol=2e-10, max_nodes=10000)
    if not solution.success:
        raise RuntimeError(f"concentration boundary-value solve failed: {solution.message}")

    beta = slip / radius
    drag_ratio = (1.0 + 2.0 * beta) / (1.0 + 3.0 * beta)
    radial_flow_coefficient = 1.5 * drag_ratio

    def integrand(r):
        x = (r - radius) / width
        concentration_amplitude = np.exp(-strength * (1.0 - x) ** 2) * solution.sol(x)[0]
        potential_derivative = -2.0 * strength * (1.0 - x) / width

        # r**2 * (1 - u_hat,r/cos(theta)); this is the reciprocal-theorem
        # kernel.  For zero slip it becomes
        # (r-a)**2 * (2*r+a) / (2*r), the familiar no-slip expression.
        kernel = (
            r**2
            - radial_flow_coefficient * radius * r
            - (1.0 - radial_flow_coefficient) * radius**3 / r
        )
        return kernel * concentration_amplitude * potential_derivative

    integral, _ = quad(integrand, radius, layer_edge, epsabs=2e-11, epsrel=2e-11)
    return 2.0 * integral / (9.0 * radius * drag_ratio)


def predict_at(experiments, viscosity):
    """Evaluate an iterable of experiment dictionaries at a given viscosity."""
    viscosity = float(viscosity)
    if not np.isfinite(viscosity) or viscosity <= 0.0:
        raise ValueError("viscosity must be a finite positive number")

    experiments = list(experiments)
    values = [
        response(
            experiment["radius"],
            experiment["width"],
            experiment["strength"],
            experiment["slip"],
        )
        for experiment in experiments
    ]
    return np.asarray(values, dtype=float) / viscosity


class Model:
    def __init__(self):
        self.viscosity = None

    def fit(self, records):
        """Fit the common viscosity by weighted Gaussian least squares.

        Each measured value is ``unit_response / viscosity`` plus independent
        Gaussian noise.  Estimating inverse viscosity makes this a one-column
        weighted linear least-squares problem and avoids an iterative fit.
        The documented viscosity interval is imposed as a physical bound.
        """
        weighted_xy = 0.0
        weighted_xx = 0.0

        for record in records:
            experiment = record["input"]
            measured = float(record["value"])
            sigma = float(record["sigma"])
            if not np.isfinite(measured) or not np.isfinite(sigma) or sigma <= 0.0:
                raise ValueError("calibration values must be finite and sigma must be positive")

            unit_response = response(
                experiment["radius"],
                experiment["width"],
                experiment["strength"],
                experiment["slip"],
            )
            weight = 1.0 / sigma**2
            weighted_xy += weight * unit_response * measured
            weighted_xx += weight * unit_response**2

        if not np.isfinite(weighted_xx) or weighted_xx <= 0.0:
            raise ValueError("calibration records contain no viscosity information")

        inverse_viscosity = weighted_xy / weighted_xx
        inverse_viscosity = np.clip(inverse_viscosity, 1.0 / 1.6, 1.0 / 0.8)
        self.viscosity = float(1.0 / inverse_viscosity)
        return self

    def predict(self, experiments):
        if self.viscosity is None:
            raise RuntimeError("fit must be called before predict")
        return predict_at(experiments, self.viscosity)
