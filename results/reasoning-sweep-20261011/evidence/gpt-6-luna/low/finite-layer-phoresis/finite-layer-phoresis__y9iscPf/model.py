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
        """Infer the common viscosity from independently measured responses.

        For each preparation the response is ``a_i / viscosity``, where
        ``a_i`` is fixed by its geometry and potential.  With independent
        Gaussian measurement errors, the maximum-likelihood estimate is the
        weighted least-squares fit of this one-parameter model.
        """
        records = list(records)
        if not records:
            raise ValueError('At least one calibration record is required.')

        coefficients = np.asarray([
            response(r['input']['radius'], r['input']['width'],
                     r['input']['strength'], r['input']['slip'])
            for r in records
        ], dtype=float)
        values = np.asarray([r['value'] for r in records], dtype=float)
        sigma = np.asarray([r['sigma'] for r in records], dtype=float)
        if (not np.all(np.isfinite(coefficients)) or
                not np.all(np.isfinite(values)) or
                not np.all(np.isfinite(sigma)) or np.any(sigma <= 0)):
            raise ValueError('Calibration values and uncertainties must be finite, with positive uncertainties.')

        # Set q = 1 / viscosity.  The weighted linear fit through the origin
        # has q = sum(w*a*y) / sum(w*a*a).
        weights = 1.0 / sigma**2
        denominator = np.sum(weights * coefficients**2)
        if denominator <= 0:
            raise ValueError('Calibration records do not identify viscosity.')
        inverse_viscosity = np.sum(weights * coefficients * values) / denominator
        if not np.isfinite(inverse_viscosity) or inverse_viscosity <= 0:
            raise ValueError('Calibration records imply an invalid viscosity.')
        self.viscosity = float(1.0 / inverse_viscosity)
        return self

    def predict(self, experiments):
        return predict_at(experiments, self.viscosity)
