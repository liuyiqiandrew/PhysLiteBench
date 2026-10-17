"""Linear diffusiophoretic response of a sphere with Navier slip."""

from functools import lru_cache

import numpy as np
from scipy.integrate import quad, solve_bvp


@lru_cache(512)
def _interaction_moments(radius, width, strength):
    """Return the two radial force moments, independent of viscosity and slip."""
    if strength == 0:
        return 0.0, 0.0

    outer = radius + width

    # Write the concentration perturbation as G exp(-U) f(r) cos(theta).
    # Zero normal solute flux gives f'(radius) = 0. Outside the potential,
    # f = r + constant/r**2, which gives the Robin condition at outer.
    def equations(x, y):
        r = radius + width * x
        return np.array((
            y[1],
            -(2 * width / r + 2 * strength * (1 - x)) * y[1]
            + 2 * width**2 * y[0] / r**2,
        ))

    def boundary(left, right):
        return np.array((left[1], outer * right[1] / width
                         + 2 * right[0] - 3 * outer))

    grid = np.linspace(0.0, 1.0, 129)
    r = radius + width * grid
    initial = np.array((r + radius**3 / (2 * r**2),
                        width * (1 - radius**3 / r**3)))
    solution = solve_bvp(equations, boundary, grid, initial,
                         tol=2e-10, max_nodes=10000)
    if not solution.success:
        raise RuntimeError(solution.message)

    def force_density(r):
        x = (r - radius) / width
        concentration = np.exp(-strength * (1 - x)**2) * solution.sol(x)[0]
        potential_derivative = -2 * strength * (1 - x) / width
        return concentration * potential_derivative

    def no_slip_moment(r):
        z = r - radius
        return z**2 * (2 * r + radius) / (2 * r) * force_density(r)

    def slip_moment(r):
        z = r - radius
        return 3 * r * z * force_density(r)

    integration_options = dict(epsabs=2e-11, epsrel=2e-11)
    return (quad(no_slip_moment, radius, outer, **integration_options)[0],
            quad(slip_moment, radius, outer, **integration_options)[0])


def response(radius, width, strength, slip):
    """Response times viscosity for one experiment."""
    no_slip, slip_part = _interaction_moments(radius, width, strength)
    beta = slip / radius
    # Reciprocal Stokes flow around a translating Navier-slip sphere gives
    # this force kernel and drag factor. The concentration field is unchanged
    # by slip in the rapid-diffusion limit.
    return 2 * (no_slip + beta * slip_part) / (9 * radius * (1 + 2 * beta))


def predict_at(experiments, viscosity):
    return np.array([response(e['radius'], e['width'], e['strength'], e['slip'])
                     for e in experiments], dtype=float) / viscosity


class Model:
    def __init__(self):
        self.viscosity = None

    def fit(self, records):
        records = list(records)
        if not records:
            raise ValueError('At least one calibration record is required.')

        response_at_unit_viscosity = predict_at(
            [record['input'] for record in records], 1.0)
        values = np.asarray([record['value'] for record in records], dtype=float)
        sigma = np.asarray([record['sigma'] for record in records], dtype=float)
        if np.any(sigma <= 0) or not np.all(np.isfinite(sigma)):
            raise ValueError('Calibration uncertainties must be positive and finite.')

        weighted_response = response_at_unit_viscosity / sigma
        information = np.dot(weighted_response, weighted_response)
        if information == 0:
            raise ValueError('Calibration records do not determine viscosity.')
        inverse_viscosity = np.dot(weighted_response, values / sigma) / information
        self.viscosity = float(1 / np.clip(inverse_viscosity, 1 / 1.6, 1 / 0.8))
        return self

    def predict(self, experiments):
        if self.viscosity is None:
            raise RuntimeError('Fit the model before predicting.')
        return predict_at(experiments, self.viscosity)
