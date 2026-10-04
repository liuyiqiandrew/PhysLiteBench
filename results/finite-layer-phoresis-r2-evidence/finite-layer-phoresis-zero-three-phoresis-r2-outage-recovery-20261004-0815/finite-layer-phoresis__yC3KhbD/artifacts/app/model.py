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
        """Fit the one unknown material parameter from calibration records.

        For a fixed experiment all of the concentration/flow physics is in
        ``response``.  The Newtonian stress makes the measured response
        proportional to the inverse viscosity, so the calibration model is

            value = response(input) / viscosity.

        This is a one-parameter weighted least-squares problem.  Fitting the
        inverse viscosity first keeps it linear and uses the supplied
        independent Gaussian uncertainties in the appropriate way.
        """
        raw = []
        values = []
        sigmas = []
        for record in records:
            experiment = record['input']
            raw.append(response(experiment['radius'], experiment['width'],
                                experiment['strength'], experiment['slip']))
            values.append(record['value'])
            sigmas.append(record['sigma'])

        raw = np.asarray(raw, dtype=float)
        values = np.asarray(values, dtype=float)
        sigmas = np.asarray(sigmas, dtype=float)
        if raw.size == 0:
            raise ValueError('at least one calibration record is required')
        if (not np.isfinite(raw).all() or not np.isfinite(values).all()
                or not np.isfinite(sigmas).all() or (sigmas <= 0).any()):
            raise ValueError('calibration values and uncertainties must be finite; '
                             'uncertainties must be positive')

        weights = 1.0 / sigmas**2
        inverse_viscosity = np.sum(weights * raw * values) / np.sum(weights * raw**2)
        if not np.isfinite(inverse_viscosity) or inverse_viscosity <= 0:
            raise ValueError('calibration records do not identify a positive viscosity')

        # The documented apparatus restricts viscosity to this interval.  A
        # small amount of measurement noise should not produce an out-of-range
        # estimate at the boundary.
        fitted = 1.0 / inverse_viscosity
        self.viscosity = float(np.clip(fitted, 0.8, 1.6))
        return self

    def predict(self, experiments):
        return predict_at(experiments, self.viscosity)
