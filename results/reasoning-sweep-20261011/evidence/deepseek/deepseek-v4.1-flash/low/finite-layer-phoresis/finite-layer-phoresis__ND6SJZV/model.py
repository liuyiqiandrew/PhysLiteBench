from functools import lru_cache

import numpy as np
from scipy.integrate import quad, solve_bvp


def _surface_drag_factor(beta):
    """Tangential-slip correction to the translational drag of a sphere.

    beta = slip / radius.  beta = 0 recovers the no-slip factor 1;
    beta -> inf recovers the perfect-slip factor 2/3.
    """
    return (1.0 + 2.0 * beta) / (1.0 + 3.0 * beta)


@lru_cache(maxsize=4096)
def response(radius, width, strength, slip):
    """Signed velocity response per unit concentration gradient.

    Solves the steady, advection-free solute transport problem in the
    frame of the sphere, then projects the resulting osmotic/interface
    force onto the translational mobility set by the slip-corrected drag.
    """
    if strength == 0.0:
        return 0.0

    end = radius + width

    def equations(x, y):
        r = radius + width * x
        return np.array([
            y[1],
            -(2.0 * width / r + 2.0 * strength * (1.0 - x)) * y[1]
            + 2.0 * width ** 2 * y[0] / r ** 2,
        ])

    def boundary(left, right):
        return np.array([left[1], end * right[1] / width + 2.0 * right[0] - 3.0 * end])

    solution = None
    for nodes in (129, 257, 513, 1025):
        grid = np.linspace(0.0, 1.0, nodes)
        r = radius + width * grid
        initial = np.array([r + radius ** 3 / (2.0 * r ** 2),
                            width * (1.0 - radius ** 3 / r ** 3)])
        trial = solve_bvp(equations, boundary, grid, initial, tol=1e-10,
                          max_nodes=20000)
        if trial.success and np.all(np.isfinite(trial.sol(x=grid))):
            solution = trial
            break
    if solution is None:
        raise RuntimeError("solute transport BVP did not converge")

    def integrand(r):
        x = (r - radius) / width
        concentration = np.exp(-strength * (1.0 - x) ** 2) * solution.sol(x)[0]
        derivative = -2.0 * strength * (1.0 - x) / width
        weight = (r - radius) ** 2 * (2.0 * r + radius) / (2.0 * r)
        return weight * concentration * derivative

    integral = quad(integrand, radius, end, epsabs=1e-12, epsrel=1e-12,
                    limit=200)[0]

    drag = _surface_drag_factor(slip / radius)
    return 2.0 / (9.0 * radius * drag) * integral


def predict_at(experiments, viscosity):
    return np.array(
        [response(e["radius"], e["width"], e["strength"], e["slip"])
         for e in experiments], dtype=float) / viscosity


class Model:
    def __init__(self):
        self.viscosity = None

    def fit(self, records):
        responses = np.array(
            [response(r["input"]["radius"], r["input"]["width"],
                      r["input"]["strength"], r["input"]["slip"])
             for r in records], dtype=float)
        values = np.array([r["value"] for r in records], dtype=float)
        sigmas = np.array([r.get("sigma", 1.0) for r in records], dtype=float)

        weights = 1.0 / sigmas ** 2
        denominator = np.sum(responses ** 2 * weights)
        if denominator <= 0.0:
            raise ValueError("calibration records carry no response signal")
        self.viscosity = float(np.sum(responses * values * weights) / denominator)
        return self

    def predict(self, experiments):
        if self.viscosity is None:
            raise RuntimeError("Model must be fit before predicting.")
        return predict_at(experiments, self.viscosity)
