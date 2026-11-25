"""Linear diffusiophoretic response of a sphere with Navier slip."""

from functools import lru_cache

import numpy as np
from scipy.integrate import solve_ivp


_QUAD_X, _QUAD_W = np.polynomial.legendre.leggauss(96)
_QUAD_X = (_QUAD_X + 1.0) / 2.0
_QUAD_W = _QUAD_W / 2.0


@lru_cache(maxsize=4096)
def response(radius, width, strength, slip):
    """Velocity per concentration gradient, multiplied by viscosity.

    The concentration perturbation is G exp(-U) f(r) cos(theta).  Its
    no-flux condition gives f'(radius)=0; outside the potential f=r+B/r**2.
    The remaining integral follows from the Stokes reciprocal theorem using
    a translating Navier-slip sphere as the auxiliary flow.
    """
    if strength == 0:
        return 0.0

    end = radius + width

    def equations(x, y):
        r = radius + width * x
        return (
            y[1],
            -(2.0 * width / r + 2.0 * strength * (1.0 - x)) * y[1]
            + 2.0 * width**2 * y[0] / r**2,
        )

    solution = solve_ivp(
        equations, (0.0, 1.0), (radius, 0.0), method="DOP853",
        rtol=1e-11, atol=1e-12, dense_output=True,
    )
    if not solution.success:
        raise RuntimeError(solution.message)

    at_end = solution.y[:, -1]
    scale = 3.0 * end / (end * at_end[1] / width + 2.0 * at_end[0])
    f = scale * solution.sol(_QUAD_X)[0]

    r = radius + width * _QUAD_X
    z = width * _QUAD_X
    beta = slip / radius
    # The reciprocal-theorem radial-flow weight after canceling the
    # translating sphere's (1 + 3*beta) drag denominator.
    kernel = (z**2 * (2.0 * r + radius) / (2.0 * r)
              + 3.0 * beta * r * z)
    integral = np.dot(
        _QUAD_W,
        kernel * np.exp(-strength * (1.0 - _QUAD_X)**2)
        * f * (-2.0 * strength * (1.0 - _QUAD_X)),
    )
    return float(2.0 * integral / (9.0 * radius * (1.0 + 2.0 * beta)))


def predict_at(experiments, viscosity):
    return np.asarray(
        [response(e["radius"], e["width"], e["strength"], e["slip"])
         for e in experiments], dtype=float,
    ) / viscosity


class Model:
    def __init__(self):
        self.viscosity = None

    def fit(self, records):
        records = list(records)
        basis = predict_at((record["input"] for record in records), 1.0)
        measured = np.asarray([record["value"] for record in records], dtype=float)
        sigma = np.asarray([record["sigma"] for record in records], dtype=float)
        weights = 1.0 / sigma**2
        denominator = np.dot(weights, basis**2)
        if denominator <= 0 or not np.isfinite(denominator):
            raise ValueError("Calibration records do not determine viscosity")
        inverse_viscosity = np.dot(weights, basis * measured) / denominator
        if inverse_viscosity <= 0 or not np.isfinite(inverse_viscosity):
            raise ValueError("Calibration records imply an invalid viscosity")
        self.viscosity = float(np.clip(1.0 / inverse_viscosity, 0.8, 1.6))
        return self

    def predict(self, experiments):
        if self.viscosity is None:
            raise ValueError("Fit the model before predicting")
        return predict_at(experiments, self.viscosity)
