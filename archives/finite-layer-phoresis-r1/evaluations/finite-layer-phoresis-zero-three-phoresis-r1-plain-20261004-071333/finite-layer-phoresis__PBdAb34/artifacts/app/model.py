from functools import lru_cache
import numpy as np
from scipy.integrate import quad, solve_bvp


@lru_cache(128)
def response(kind, width, strength, radius=1.):
    if strength == 0:
        return 0.
    if kind == 'wall':
        return -quad(lambda z: z*np.expm1(-strength*(1-z/width)**2),
                     0., width, epsabs=1e-11, epsrel=1e-11)[0]
    end = radius+width
    def potential(r):
        return strength*((end-r)/width)**2
    def derivative(r):
        return -2*strength*(end-r)/width**2
    def equations(r, y):
        return np.array([y[1], -(2/r-derivative(r))*y[1]+2*y[0]/r**2])
    def boundary(left, right):
        return np.array([left[1], end*right[1]+2*right[0]-3*end])
    grid = np.linspace(radius, end, 129)
    initial = np.array([grid+radius**3/(2*grid**2), 1-radius**3/grid**3])
    solution = solve_bvp(equations, boundary, grid, initial, tol=2e-9, max_nodes=10000)
    if not solution.success:
        raise RuntimeError(solution.message)
    def integrand(r):
        concentration = np.exp(-potential(r))*solution.sol(r)[0]
        # For a translating sphere the radial component of the auxiliary
        # Stokes flow is
        #   1 - 3R/(2r) + R**3/(2r**3).
        # The reciprocal-theorem force weight is its complement, multiplied
        # by the spherical volume factor r**2.  It reduces to 3*z**2/2 only
        # in the thin interaction-layer limit; the stated model also allows
        # widths comparable to (or larger than) the sphere radius.
        weight = r**2*(1. - 1.5*radius/r + 0.5*(radius/r)**3)
        return weight*concentration*derivative(r)
    return 2/(9*radius)*quad(integrand, radius, end, epsabs=2e-10, epsrel=2e-10)[0]


def predict_at(experiments, viscosity):
    return np.array([response(e['kind'], e['width'], e['strength'], e.get('radius', 1.))
                     for e in experiments])/viscosity


class Model:
    def __init__(self):
        self.viscosity = None

    def fit(self, records):
        """Fit the (single) viscosity parameter to calibration records.

        For a given experiment the hydrodynamic response is independent of
        viscosity, and the measured velocity is that response divided by
        viscosity.  Thus the records are a weighted linear regression in
        ``x = 1 / viscosity``.  Using the supplied uncertainties is
        important here: they are the instrument uncertainties, rather than
        uncertainties to be estimated from the calibration set.
        """
        records = list(records)
        if not records:
            raise ValueError('At least one calibration record is required.')

        calculated = []
        measured = []
        weights = []
        for record in records:
            experiment = record['input']
            sigma = float(record['sigma'])
            if not np.isfinite(sigma) or sigma <= 0.0:
                raise ValueError('Calibration uncertainties must be positive.')
            calculated.append(response(
                experiment['kind'],
                float(experiment['width']),
                float(experiment['strength']),
                float(experiment.get('radius', 1.)),
            ))
            measured.append(float(record['value']))
            weights.append(1.0 / sigma**2)

        calculated = np.asarray(calculated, dtype=float)
        measured = np.asarray(measured, dtype=float)
        weights = np.asarray(weights, dtype=float)
        if not (np.isfinite(calculated).all() and np.isfinite(measured).all()):
            raise ValueError('Calibration values must be finite.')

        denominator = np.sum(weights * calculated**2)
        if not np.isfinite(denominator) or denominator <= 0.0:
            raise ValueError('Calibration records contain no model signal.')

        # Minimize sum_i ((x*a_i - y_i)/sigma_i)^2, then enforce the
        # documented physical range for viscosity.
        inverse_viscosity = np.sum(weights * calculated * measured) / denominator
        inverse_viscosity = np.clip(inverse_viscosity, 1.0 / 1.6, 1.0 / 0.8)
        self.viscosity = float(1.0 / inverse_viscosity)
        return self

    def predict(self, experiments):
        if self.viscosity is None:
            raise RuntimeError('Model must be fitted before prediction.')
        return predict_at(experiments, self.viscosity)
