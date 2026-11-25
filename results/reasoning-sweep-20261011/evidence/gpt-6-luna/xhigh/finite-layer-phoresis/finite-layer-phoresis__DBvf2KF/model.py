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
    # Radial flow in the auxiliary unit-translation problem with the same
    # Navier condition is (A/r + B/r**3) cos(theta).  The force-free
    # interaction kernel uses the relative radial flow, r**2*(1-u_r/U).
    # For zero slip this reduces to the familiar no-slip expression below.
    aux_a = 1.5*radius*drag
    aux_b = -0.5*radius**3/(1+3*beta)
    def integrand(r):
        x = (r-radius)/width
        concentration = np.exp(-strength*(1-x)**2)*solution.sol(x)[0]
        derivative = -2*strength*(1-x)/width
        weight = r**2-aux_a*r-aux_b/r
        return weight*concentration*derivative
    return 2/(9*radius*drag)*quad(integrand, radius, end, epsabs=2e-11, epsrel=2e-11)[0]


def predict_at(experiments, viscosity):
    return np.array([response(e['radius'], e['width'], e['strength'], e['slip'])
                     for e in experiments])/viscosity


class Model:
    def __init__(self):
        self.viscosity = None

    def fit(self, records):
        """Infer the common viscosity from calibrated response measurements.

        ``response`` contains the part of the prediction that depends on the
        experiment; the measured velocity gradient response is that quantity
        divided by the solvent viscosity.  Thus the inverse viscosity is a
        single linear parameter and can be fitted by weighted least squares.
        """
        records = list(records)
        if not records:
            raise ValueError('At least one calibration record is required.')

        coefficients = np.asarray(
            [response(**record['input']) for record in records], dtype=float
        )
        values = np.asarray([record['value'] for record in records], dtype=float)
        sigmas = np.asarray([record['sigma'] for record in records], dtype=float)

        if (not np.all(np.isfinite(coefficients))
                or not np.all(np.isfinite(values))
                or not np.all(np.isfinite(sigmas))
                or np.any(sigmas <= 0)):
            raise ValueError('Calibration values and uncertainties must be finite, with positive uncertainties.')

        weights = 1.0 / np.square(sigmas)
        denominator = np.sum(weights * np.square(coefficients))
        if denominator == 0 or not np.isfinite(denominator):
            raise ValueError('Calibration records do not contain information about viscosity.')

        inverse_viscosity = np.sum(weights * coefficients * values) / denominator
        # The apparatus specification bounds viscosity to [.8, 1.6].  Apply
        # the corresponding parameter bound when noisy data put the fit out
        # of range.
        inverse_viscosity = np.clip(inverse_viscosity, 1.0 / 1.6, 1.0 / 0.8)
        self.viscosity = float(1.0 / inverse_viscosity)
        return self

    def predict(self, experiments):
        if self.viscosity is None:
            raise ValueError('Call fit(records) before predict(experiments).')
        return predict_at(experiments, self.viscosity)
