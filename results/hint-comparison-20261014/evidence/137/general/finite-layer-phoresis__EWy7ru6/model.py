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
        """Infer the solvent viscosity from the calibration measurements.

        For fixed apparatus parameters the response is independent of the
        viscosity apart from the Stokes mobility factor, so each calibration
        measurement has the form

            value = response(input) / viscosity + Gaussian noise.

        Thus the weighted least-squares problem is linear in
        ``inverse_viscosity``.  Fitting that parameter also makes the use of
        the supplied, possibly repeated, calibration measurements explicit.
        """
        records = list(records)
        if not records:
            raise ValueError('At least one calibration record is required.')

        calculated = np.asarray([
            response(**record['input']) for record in records
        ], dtype=float)
        measured = np.asarray([record['value'] for record in records],
                              dtype=float)
        sigma = np.asarray([record['sigma'] for record in records],
                           dtype=float)

        if (not np.isfinite(calculated).all()
                or not np.isfinite(measured).all()
                or not np.isfinite(sigma).all()
                or np.any(sigma <= 0)):
            raise ValueError('Calibration records must contain finite values '
                             'and positive uncertainties.')

        weights = 1.0 / sigma**2
        denominator = np.sum(weights * calculated**2)
        if denominator == 0:
            raise ValueError('Calibration records contain no response signal.')

        inverse_viscosity = (np.sum(weights * calculated * measured)
                             / denominator)
        if not np.isfinite(inverse_viscosity) or inverse_viscosity <= 0:
            raise ValueError('Calibration records imply a non-positive '
                             'viscosity.')

        # The apparatus specification gives the physical prior range for the
        # sole unknown.  It is inactive for the supplied calibration data but
        # keeps the fitted model physical for unusually noisy records.
        self.viscosity = float(np.clip(1.0 / inverse_viscosity, 0.8, 1.6))
        return self

    def predict(self, experiments):
        return predict_at(experiments, self.viscosity)
