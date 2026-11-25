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
        weight = 1.5*(r-radius)**2
        return weight*concentration*derivative(r)
    return 2/(9*radius)*quad(integrand, radius, end, epsabs=2e-10, epsrel=2e-10)[0]


def predict_at(experiments, viscosity):
    return np.array([response(e['kind'], e['width'], e['strength'], e.get('radius', 1.))
                     for e in experiments])/viscosity


class Model:
    def __init__(self):
        self.viscosity = None

    def fit(self, records):
        """Fit the one unknown material parameter from calibration records.

        For a fixed interaction potential the creeping-flow response is
        proportional to the reciprocal viscosity.  Consequently the
        calibration problem is a one-parameter weighted linear least-squares
        problem in ``1 / viscosity``.
        """
        records = list(records)
        if not records:
            raise ValueError('at least one calibration record is required')

        basis = np.empty(len(records), dtype=float)
        values = np.empty(len(records), dtype=float)
        sigmas = np.empty(len(records), dtype=float)
        for i, record in enumerate(records):
            experiment = record['input']
            basis[i] = response(
                experiment['kind'],
                experiment['width'],
                experiment['strength'],
                experiment.get('radius', 1.),
            )
            values[i] = float(record['value'])
            sigmas[i] = float(record.get('sigma', 1.))

        if (not np.isfinite(basis).all() or not np.isfinite(values).all()
                or not np.isfinite(sigmas).all() or np.any(sigmas <= 0)):
            raise ValueError('calibration records must contain finite values')

        weights = 1. / sigmas**2
        denominator = np.sum(weights * basis**2)
        if denominator == 0.:
            raise ValueError('calibration records contain no informative response')
        reciprocal_viscosity = np.sum(weights * basis * values) / denominator
        if not np.isfinite(reciprocal_viscosity) or reciprocal_viscosity <= 0.:
            raise ValueError('calibration records imply an invalid viscosity')

        # The stated apparatus restricts viscosity to this interval.  The
        # unconstrained estimate is used when it is admissible; clipping gives
        # the constrained least-squares estimate for noisy/outlying records.
        self.viscosity = float(np.clip(1. / reciprocal_viscosity, .8, 1.6))
        return self

    def predict(self, experiments):
        return predict_at(experiments, self.viscosity)
