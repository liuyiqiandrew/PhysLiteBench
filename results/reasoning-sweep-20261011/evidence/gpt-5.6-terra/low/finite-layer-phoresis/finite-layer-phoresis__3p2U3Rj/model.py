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
        """Fit the (single) viscosity parameter by weighted least squares.

        The response calculated above is the velocity per gradient at unit
        viscosity.  Thus each observation is linear in the inverse
        viscosity, which makes a one-parameter weighted fit both exact and
        more stable than a numerical optimizer.
        """
        records = list(records)
        if not records:
            raise ValueError("at least one calibration record is required")

        unit_response = np.asarray(
            [response(**record["input"]) for record in records], dtype=float
        )
        values = np.asarray([record["value"] for record in records], dtype=float)
        sigma = np.asarray([record["sigma"] for record in records], dtype=float)
        if (not np.isfinite(unit_response).all() or not np.isfinite(values).all()
                or not np.isfinite(sigma).all() or np.any(sigma <= 0)):
            raise ValueError("calibration values and uncertainties must be finite; sigma must be positive")

        # y = a*x, where a=1/viscosity.  The physical parameter is positive.
        weights = 1.0 / sigma**2
        denominator = np.sum(weights * unit_response**2)
        numerator = np.sum(weights * unit_response * values)
        if not np.isfinite(denominator) or denominator <= 0 or numerator <= 0:
            raise ValueError("calibration records do not identify a positive viscosity")
        self.viscosity = float(denominator / numerator)
        return self

    def predict(self, experiments):
        if self.viscosity is None:
            raise RuntimeError("fit must be called before predict")
        return predict_at(experiments, self.viscosity)
