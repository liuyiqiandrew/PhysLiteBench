from functools import lru_cache
import numpy as np
from scipy.integrate import quad, solve_bvp


@lru_cache(256)
def response(radius, width, strength, slip):
    if strength == 0:
        return 0.
    end = radius+width
    def equations(x, y):
        r = radius+width*x
        return np.array([y[1], -(2*width/r+2*strength*(1-x))*y[1]+2*width**2*y[0]/r**2])
    def boundary(left, right):
        return np.array([left[1], end*right[1]/width+2*right[0]-3*end])
    grid = np.linspace(0., 1., 129)
    r = radius+width*grid
    initial = np.array([r+radius**3/(2*r**2), width*(1-radius**3/r**3)])
    solution = solve_bvp(equations, boundary, grid, initial, tol=2e-10, max_nodes=10000)
    if not solution.success:
        raise RuntimeError(solution.message)
    beta = slip/radius
    drag = (1+2*beta)/(1+3*beta)
    def integrand(r):
        x = (r-radius)/width
        concentration = np.exp(-strength*(1-x)**2)*solution.sol(x)[0]
        derivative = -2*strength*(1-x)/width
        weight = (r-radius)**2*(2*r+radius)/(2*r)
        return weight*concentration*derivative
    return 2/(9*radius*drag)*quad(integrand, radius, end, epsabs=2e-11, epsrel=2e-11)[0]


def predict_at(experiments, viscosity):
    return np.array([response(e['radius'], e['width'], e['strength'], e['slip'])
                     for e in experiments])/viscosity


class Model:
    def __init__(self):
        self.viscosity = None

    def fit(self, records):
        """Estimate the (single) viscosity from independent calibrations.

        ``response`` is the mobility apart from viscosity, so a calibration
        value has mean ``response(input) / viscosity``.  It is convenient to
        fit its reciprocal: this makes the Gaussian weighted least-squares
        problem a one-parameter linear regression through the origin.
        """
        records = list(records)
        if not records:
            raise ValueError("at least one calibration record is required")

        mobilities = np.asarray(
            [response(**record["input"]) for record in records], dtype=float
        )
        values = np.asarray([record["value"] for record in records], dtype=float)
        sigmas = np.asarray([record["sigma"] for record in records], dtype=float)

        if (not np.isfinite(mobilities).all() or not np.isfinite(values).all()
                or not np.isfinite(sigmas).all() or np.any(sigmas <= 0)):
            raise ValueError("calibration values and uncertainties must be finite; sigma must be positive")

        weights = 1.0 / np.square(sigmas)
        denominator = np.dot(weights, np.square(mobilities))
        if not np.isfinite(denominator) or denominator <= 0:
            raise ValueError("calibrations contain no information about viscosity")

        inverse_viscosity = np.dot(weights * mobilities, values) / denominator
        if not np.isfinite(inverse_viscosity) or inverse_viscosity <= 0:
            raise ValueError("calibrations do not imply a positive viscosity")

        # The apparatus specification supplies the physically admissible
        # interval.  For normal calibration data this is inactive, but it
        # keeps a noisy extrapolation from producing an invalid apparatus.
        self.viscosity = float(np.clip(1.0 / inverse_viscosity, 0.8, 1.6))
        return self

    def predict(self, experiments):
        if self.viscosity is None:
            raise RuntimeError("fit must be called before predict")
        return predict_at(experiments, self.viscosity)
