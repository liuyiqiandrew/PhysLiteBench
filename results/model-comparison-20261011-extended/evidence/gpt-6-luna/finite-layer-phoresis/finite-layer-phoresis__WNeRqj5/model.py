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

        The response computed by :func:`response` is the response at unit
        viscosity. Since Stokes flow is linear in the stress scale, the
        predicted response is that value divided by the uniform viscosity.
        Fit inverse viscosity as a weighted scale factor.
        """
        records = list(records)
        if not records:
            raise ValueError('At least one calibration record is required.')

        unit_response = np.asarray(
            [response(**record['input']) for record in records], dtype=float
        )
        measured = np.asarray([record['value'] for record in records], dtype=float)
        sigma = np.asarray([record['sigma'] for record in records], dtype=float)

        if (not np.isfinite(unit_response).all()
                or not np.isfinite(measured).all()
                or not np.isfinite(sigma).all()
                or np.any(sigma <= 0)):
            raise ValueError('Calibration values and positive uncertainties must be finite.')

        # Minimize sum(((alpha * response - measured) / sigma)**2), where
        # alpha = 1 / viscosity. Scaling first improves numerical stability.
        design = unit_response / sigma
        target = measured / sigma
        denominator = np.dot(design, design)
        if not np.isfinite(denominator) or denominator <= 0:
            raise ValueError('Calibration records do not constrain viscosity.')
        inverse_viscosity = np.dot(design, target) / denominator

        # Constrain the fit to the documented viscosity range [.8, 1.6].
        inverse_viscosity = np.clip(inverse_viscosity, 1.0 / 1.6, 1.0 / 0.8)
        self.viscosity = float(1.0 / inverse_viscosity)
        return self

    def predict(self, experiments):
        if self.viscosity is None:
            raise ValueError('Call fit(records) before predict(experiments).')
        return predict_at(experiments, self.viscosity)
