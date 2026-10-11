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
        """Infer the solvent viscosity from independent calibration records.

        For a fixed experiment the response is inversely proportional to the
        viscosity, so if ``q_i`` is the viscosity-independent response and
        ``y_i`` is the measured value, the model is ``y_i = q_i / mu``.
        The calibration uncertainties are independent Gaussians; minimizing
        the corresponding weighted squared residual therefore gives a
        one-dimensional closed-form estimate for ``mu``.
        """
        if not records:
            raise ValueError('at least one calibration record is required')

        experiments = [record['input'] for record in records]
        values = np.asarray([record['value'] for record in records], dtype=float)
        sigmas = np.asarray([record['sigma'] for record in records], dtype=float)
        independent = np.asarray(predict_at(experiments, 1.0), dtype=float)

        if (values.shape != independent.shape or
                not np.all(np.isfinite(values)) or
                not np.all(np.isfinite(independent)) or
                not np.all(np.isfinite(sigmas)) or
                np.any(sigmas <= 0)):
            raise ValueError('calibration records must contain finite values')

        weights = 1.0 / sigmas**2
        denominator = np.sum(weights * independent * values)
        if denominator <= 0 or not np.isfinite(denominator):
            raise ValueError('calibration records do not identify a positive viscosity')

        viscosity = np.sum(weights * independent**2) / denominator
        # The stated apparatus restricts viscosity to this interval.  A
        # bounded fit is the appropriate likelihood estimate if noisy input
        # records happen to put the unconstrained estimate just outside it.
        self.viscosity = float(np.clip(viscosity, 0.8, 1.6))
        return self

    def predict(self, experiments):
        if self.viscosity is None:
            raise RuntimeError('fit must be called before predict')
        result = np.asarray(predict_at(experiments, self.viscosity), dtype=float)
        if result.ndim != 1 or not np.all(np.isfinite(result)):
            raise ValueError('prediction produced non-finite values')
        return result
