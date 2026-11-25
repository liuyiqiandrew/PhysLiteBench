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
        """Infer the solvent viscosity from calibration measurements.

        For fixed apparatus parameters the response is proportional to the
        inverse viscosity.  Writing the response as ``q * response0``, where
        ``q = 1 / viscosity``, makes the Gaussian maximum-likelihood fit a
        one-parameter weighted linear least-squares problem.
        """
        records = list(records)
        if not records:
            raise ValueError('At least one calibration record is required.')

        response0 = np.asarray([
            response(**record['input']) for record in records
        ], dtype=float)
        values = np.asarray([record['value'] for record in records],
                            dtype=float)
        sigma = np.asarray([record['sigma'] for record in records],
                           dtype=float)

        if (not np.isfinite(response0).all() or
                not np.isfinite(values).all() or
                not np.isfinite(sigma).all() or
                np.any(sigma <= 0)):
            raise ValueError('Calibration records must contain finite values '
                             'and positive uncertainties.')

        weights = 1.0 / sigma**2
        denominator = np.sum(weights * response0**2)
        if denominator == 0.0:
            raise ValueError('Calibration records contain no viscosity '
                             'information.')

        # Fit q = 1 / viscosity, then apply the stated physical bounds.
        inverse_viscosity = np.sum(weights * response0 * values) / denominator
        inverse_viscosity = np.clip(inverse_viscosity, 1.0 / 1.6, 1.0 / .8)
        self.viscosity = float(1.0 / inverse_viscosity)
        return self

    def predict(self, experiments):
        if self.viscosity is None:
            raise RuntimeError('Model must be fitted before prediction.')
        return predict_at(experiments, self.viscosity)
