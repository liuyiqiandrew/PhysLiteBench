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
        """Infer the common viscosity from independent calibration records.

        The response is inversely proportional to viscosity, so the weighted
        least-squares estimate has a closed form.  Writing the model response
        as ``a_i / viscosity``, minimize sum(((a_i / viscosity-y_i)/sigma_i)**2).
        """
        records = list(records)
        if not records:
            raise ValueError('At least one calibration record is required.')

        experiments = [record['input'] for record in records]
        measured = np.asarray([record['value'] for record in records], dtype=float)
        sigma = np.asarray([record.get('sigma', 1.0) for record in records], dtype=float)
        if np.any(~np.isfinite(measured)) or np.any(~np.isfinite(sigma)) or np.any(sigma <= 0):
            raise ValueError('Calibration values and uncertainties must be finite, with positive uncertainties.')

        unit_viscosity_response = predict_at(experiments, 1.0)
        weights = 1.0 / sigma**2
        numerator = np.sum(weights * unit_viscosity_response**2)
        denominator = np.sum(weights * unit_viscosity_response * measured)
        if not np.isfinite(denominator) or denominator <= 0:
            raise ValueError('Calibration records do not identify a positive viscosity.')
        viscosity = numerator / denominator
        if not np.isfinite(viscosity) or viscosity <= 0:
            raise ValueError('Calibration records do not identify a positive viscosity.')
        self.viscosity = float(viscosity)
        return self

    def predict(self, experiments):
        return predict_at(experiments, self.viscosity)
