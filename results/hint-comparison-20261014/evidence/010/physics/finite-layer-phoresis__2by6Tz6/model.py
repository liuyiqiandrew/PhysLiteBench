"""Linear diffusiophoretic response of a sphere with a Navier boundary.

The concentration problem is expressed through ``q = c exp(U)``.  Its
dipolar coefficient is found with a one-dimensional boundary-value solve;
the velocity follows from the reciprocal theorem including the reaction of
the potential carried by the sphere.
"""

from functools import lru_cache

import numpy as np
from scipy.integrate import quad, solve_bvp


@lru_cache(maxsize=512)
def response(radius, width, strength, slip):
    """Return the response multiplied by viscosity."""
    radius, width, strength, slip = map(float, (radius, width, strength, slip))
    if strength == 0.0:
        return 0.0
    outer = radius + width

    # q/G = h(r) cos(theta), with div(exp(-U) grad(q)) = 0.
    def equations(x, y):
        r = radius + width * x
        return np.vstack((
            y[1],
            -(2.0 * width / r + 2.0 * strength * (1.0 - x)) * y[1]
            + 2.0 * width ** 2 * y[0] / r ** 2,
        ))

    # h'(a)=0; outside the layer h=r+C/r^2.
    def boundary(left, right):
        return np.array((left[1], outer * right[1] / width + 2.0 * right[0] - 3.0 * outer))

    grid = np.linspace(0.0, 1.0, 129)
    r_grid = radius + width * grid
    initial = np.vstack((
        r_grid + radius ** 3 / (2.0 * r_grid ** 2),
        width * (1.0 - radius ** 3 / r_grid ** 3),
    ))
    solution = solve_bvp(equations, boundary, grid, initial, tol=2e-10, max_nodes=10000)
    if not solution.success:
        raise RuntimeError("concentration boundary-value solve failed: " + solution.message)

    # The unit translating auxiliary sphere obeys the *same* Navier condition:
    # uhat_r/cos(theta) = A/r+B/r^3, with the dimensionless coefficients below.
    beta = slip / radius
    denominator = 1.0 + 3.0 * beta
    a_over = 3.0 * (1.0 + 2.0 * beta) / (2.0 * denominator)  # A/radius
    b_over = -1.0 / (2.0 * denominator)                       # B/radius**3
    drag_factor = (1.0 + 2.0 * beta) / denominator

    def integrand(r):
        x = (r - radius) / width
        potential = strength * (1.0 - x) ** 2
        h = solution.sol(x)[0]
        d_potential = -2.0 * strength * (1.0 - x) / width
        # r^2[1-uhat_r/cos(theta)].  The 1 is the potential reaction on
        # the particle; omitting it would retain only the fluid body force.
        reciprocal_weight = r ** 2 - a_over * radius * r - b_over * radius ** 3 / r
        return np.exp(-potential) * h * d_potential * reciprocal_weight

    integral = quad(integrand, radius, outer, epsabs=2e-11, epsrel=2e-11, limit=150)[0]
    return 2.0 * integral / (9.0 * radius * drag_factor)


def predict_at(experiments, viscosity):
    """Evaluate the documented response for an iterable of input mappings."""
    if not np.isfinite(viscosity) or viscosity <= 0.0:
        raise ValueError("viscosity must be a positive finite number")
    return np.asarray(
        [response(e["radius"], e["width"], e["strength"], e["slip"]) for e in experiments],
        dtype=float,
    ) / viscosity


class Model:
    def __init__(self):
        self.viscosity = None

    def fit(self, records):
        """Infer viscosity by Gaussian weighted least squares and return self."""
        records = list(records)
        if not records:
            raise ValueError("at least one calibration record is required")
        coefficients = np.asarray(
            [response(r["input"]["radius"], r["input"]["width"],
                      r["input"]["strength"], r["input"]["slip"])
             for r in records], dtype=float)
        observed = np.asarray([r["value"] for r in records], dtype=float)
        sigma = np.asarray([r["sigma"] for r in records], dtype=float)
        if (not np.isfinite(observed).all() or not np.isfinite(sigma).all()
                or np.any(sigma <= 0.0)):
            raise ValueError("calibration values must be finite and sigmas positive")

        # y_i = coefficient_i / viscosity is linear in inverse viscosity.
        weights = 1.0 / sigma ** 2
        normalizer = np.sum(weights * coefficients ** 2)
        if not np.isfinite(normalizer) or normalizer <= 0.0:
            raise ValueError("calibration contains no viscosity-sensitive response")
        inverse_viscosity = np.sum(weights * coefficients * observed) / normalizer
        if not np.isfinite(inverse_viscosity) or inverse_viscosity <= 0.0:
            raise ValueError("calibration does not determine a positive viscosity")
        # The stated apparatus restricts the sole fitted material parameter.
        self.viscosity = float(np.clip(1.0 / inverse_viscosity, 0.8, 1.6))
        return self

    def predict(self, experiments):
        if self.viscosity is None:
            raise RuntimeError("fit must be called before predict")
        return predict_at(experiments, self.viscosity)
