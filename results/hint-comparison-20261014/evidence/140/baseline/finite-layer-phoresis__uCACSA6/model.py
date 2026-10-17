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
        """Estimate the (common) viscosity from independent measurements.

        For a given preparation the response calculated by ``response`` is
        inversely proportional to viscosity.  Writing that relationship as
        ``value = base * q`` with ``q = 1 / viscosity`` makes the Gaussian
        maximum-likelihood estimate a one-parameter weighted least-squares
        problem.  This is both more accurate than averaging per-record
        viscosity estimates and remains well behaved for weak interactions.
        """
        records = list(records)
        if not records:
            raise ValueError("at least one calibration record is required")

        bases = []
        values = []
        weights = []
        for record in records:
            try:
                experiment = record["input"]
                value = float(record["value"])
                sigma = float(record["sigma"])
                base = response(
                    float(experiment["radius"]),
                    float(experiment["width"]),
                    float(experiment["strength"]),
                    float(experiment["slip"]),
                )
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError("invalid calibration record") from error

            if not (np.isfinite(value) and np.isfinite(sigma) and sigma > 0
                    and np.isfinite(base)):
                raise ValueError("calibration values and uncertainties must be finite; sigma must be positive")
            bases.append(base)
            values.append(value)
            weights.append(1.0 / sigma**2)

        bases = np.asarray(bases, dtype=float)
        values = np.asarray(values, dtype=float)
        weights = np.asarray(weights, dtype=float)
        denominator = np.dot(weights, bases * bases)
        if not np.isfinite(denominator) or denominator <= 0:
            raise ValueError("calibration records do not identify viscosity")

        inverse_viscosity = np.dot(weights, bases * values) / denominator
        # The stated apparatus constrains the unknown to this interval.  The
        # constrained MLE is therefore the clipped estimate in q-space.
        inverse_viscosity = np.clip(inverse_viscosity, 1.0 / 1.6, 1.0 / 0.8)
        self.viscosity = float(1.0 / inverse_viscosity)
        return self

    def predict(self, experiments):
        if self.viscosity is None:
            raise RuntimeError("fit must be called before predict")
        return predict_at(experiments, self.viscosity)
