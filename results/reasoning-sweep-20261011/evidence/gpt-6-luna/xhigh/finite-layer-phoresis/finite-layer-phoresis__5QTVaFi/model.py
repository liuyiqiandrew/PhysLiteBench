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
        """Infer the viscosity from independent calibration measurements.

        The hydrodynamic response is linear in inverse viscosity, so each
        record has the form ``value = response(input) / viscosity``.  The
        reported instrument uncertainties are independent Gaussian standard
        deviations; use them as weights in the least-squares estimate.
        """
        records = list(records)
        if not records:
            raise ValueError('At least one calibration record is required.')

        responses = np.asarray(
            [response(**record['input']) for record in records], dtype=float
        )
        values = np.asarray([record['value'] for record in records], dtype=float)
        sigmas = np.asarray([record['sigma'] for record in records], dtype=float)

        if (not np.all(np.isfinite(responses))
                or not np.all(np.isfinite(values))
                or not np.all(np.isfinite(sigmas))
                or np.any(sigmas <= 0)):
            raise ValueError('Calibration values, responses, and uncertainties must be finite; uncertainties must be positive.')

        # Fit q = 1 / viscosity in the linear model value = q * response.
        weights = 1.0 / sigmas**2
        denominator = np.sum(weights * responses**2)
        if denominator == 0:
            raise ValueError('Calibration records do not identify the viscosity.')
        inverse_viscosity = np.sum(weights * responses * values) / denominator
        if not np.isfinite(inverse_viscosity):
            raise ValueError('Calibration records imply an invalid viscosity.')

        # Constrain the fit to the stated viscosity interval [.8, 1.6].
        inverse_viscosity = np.clip(inverse_viscosity, 1.0 / 1.6, 1.0 / 0.8)
        self.viscosity = float(1.0 / inverse_viscosity)
        return self

    def predict(self, experiments):
        return predict_at(experiments, self.viscosity)
