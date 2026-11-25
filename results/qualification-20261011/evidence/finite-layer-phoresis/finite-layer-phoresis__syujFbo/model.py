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

        For a fixed experiment all of the response physics is independent of
        viscosity; viscosity only scales the response as ``1 / viscosity``.
        Thus, writing ``q`` for the viscosity-free response and ``a`` for its
        reciprocal viscosity, the calibration model is the weighted linear
        model

            value = a * q + noise.

        Solving that one-parameter weighted least-squares problem analytically
        avoids an unnecessary numerical optimizer and uses the supplied
        Gaussian uncertainties correctly.
        """
        records = list(records)
        if not records:
            raise ValueError('At least one calibration record is required.')

        responses = np.asarray(
            [response(**record['input']) for record in records], dtype=float
        )
        values = np.asarray([record['value'] for record in records], dtype=float)
        sigmas = np.asarray([record['sigma'] for record in records], dtype=float)

        if (not np.isfinite(responses).all() or
                not np.isfinite(values).all() or
                not np.isfinite(sigmas).all() or
                np.any(sigmas <= 0)):
            raise ValueError('Calibration values and uncertainties must be finite; '
                             'uncertainties must be positive.')

        weights = 1.0 / sigmas**2
        denominator = np.sum(weights * responses * values)
        numerator = np.sum(weights * responses**2)
        if numerator == 0.0 or denominator <= 0.0:
            raise ValueError('Calibration records do not identify a positive viscosity.')

        # The fit is linear in reciprocal viscosity.  The stated parameter
        # range supplies a useful guard against an unphysical noisy fit.
        reciprocal_viscosity = denominator / numerator
        viscosity = 1.0 / reciprocal_viscosity
        if not 0.8 <= viscosity <= 1.6:
            raise ValueError('Inferred viscosity is outside the documented range.')
        self.viscosity = float(viscosity)
        return self

    def predict(self, experiments):
        return predict_at(experiments, self.viscosity)
