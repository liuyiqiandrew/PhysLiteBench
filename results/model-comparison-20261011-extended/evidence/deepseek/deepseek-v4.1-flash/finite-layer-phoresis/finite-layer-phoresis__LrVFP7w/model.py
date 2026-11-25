from functools import lru_cache

import numpy as np
from scipy.integrate import quad, solve_bvp

# Physical bounds documented for the solvent viscosity.
VISCOSITY_LOW = 0.8
VISCOSITY_HIGH = 1.6


@lru_cache(512)
def response(radius, width, strength, slip):
    """Signed steady translation per unit far-field gradient G.

    Returns the response *before* dividing by the (a priori unknown) solvent
    viscosity, i.e. predict = response(radius, width, strength, slip) / viscosity.
    """
    if strength == 0:
        return 0.
    end = radius + width

    def equations(x, y):
        r = radius + width * x
        return np.array([
            y[1],
            -(2 * width / r + 2 * strength * (1 - x)) * y[1] + 2 * width ** 2 * y[0] / r ** 2,
        ])

    def boundary(left, right):
        return np.array([left[1], end * right[1] / width + 2 * right[0] - 3 * end])

    grid = np.linspace(0., 1., 129)
    r = radius + width * grid
    initial = np.array([r + radius ** 3 / (2 * r ** 2), width * (1 - radius ** 3 / r ** 3)])
    solution = solve_bvp(equations, boundary, grid, initial, tol=2e-10, max_nodes=10000)
    if not solution.success:
        raise RuntimeError(solution.message)

    beta = slip / radius
    drag = (1 + 2 * beta) / (1 + 3 * beta)

    def integrand(r):
        x = (r - radius) / width
        concentration = np.exp(-strength * (1 - x) ** 2) * solution.sol(x)[0]
        derivative = -2 * strength * (1 - x) / width
        weight = (r - radius) ** 2 * (2 * r + radius) / (2 * r)
        return weight * concentration * derivative

    return 2 / (9 * radius * drag) * quad(integrand, radius, end, epsabs=2e-11, epsrel=2e-11)[0]


def predict_at(experiments, viscosity):
    if viscosity is None:
        raise ValueError('Model must be fitted before prediction; call Model.fit first.')
    return np.array(
        [response(e['radius'], e['width'], e['strength'], e['slip']) for e in experiments],
        dtype=float,
    ) / viscosity


class Model:
    def __init__(self):
        self.viscosity = None

    def fit(self, records):
        records = list(records)
        R = np.array(
            [response(r['input']['radius'], r['input']['width'],
                      r['input']['strength'], r['input']['slip']) for r in records],
            dtype=float,
        )
        V = np.array([r['value'] for r in records], dtype=float)
        S = np.array([r['sigma'] for r in records], dtype=float)

        if len(records) == 0:
            raise ValueError('No calibration records supplied.')

        weights = 1.0 / S ** 2
        # value ~ response / viscosity  =>  fit response = viscosity * value in
        # a weighted least-squares sense, giving the closed-form minimiser below.
        denominator = float(np.sum(weights * R * V))
        if np.isfinite(denominator) and denominator != 0.0:
            estimate = float(np.sum(weights * R * R) / denominator)
        else:
            estimate = 0.5 * (VISCOSITY_LOW + VISCOSITY_HIGH)

        if not np.isfinite(estimate) or estimate <= 0.0:
            estimate = 0.5 * (VISCOSITY_LOW + VISCOSITY_HIGH)

        self.viscosity = float(np.clip(estimate, VISCOSITY_LOW, VISCOSITY_HIGH))
        return self

    def predict(self, experiments):
        return predict_at(experiments, self.viscosity)
