from functools import lru_cache
import numpy as np
from scipy.integrate import quad, solve_bvp


@lru_cache(256)
def response(radius, width, strength, slip):
    """Return the response per unit viscosity for one experiment.

    The concentration problem is written for ``c = exp(-U) f``.  The
    coefficient of the l=1 part of ``f`` is the solution of the radial
    diffusion equation below.  The force calculation is a reciprocal
    theorem calculation using a translating auxiliary sphere.  In
    particular, the ``1`` in ``1-u_r_star/U`` is the reaction on the sphere
    of the carried solute potential; it must not be dropped.
    """
    radius = float(radius)
    width = float(width)
    strength = float(strength)
    slip = float(slip)
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

    # For a unit translating sphere, write the decaying auxiliary flow as
    #
    #   u_r_star = cos(theta) * (A/r + B/r**3).
    #
    # Impermeability gives A/radius + B/radius**3 = 1.  Applying the stated
    # Navier condition to the tangential stress gives the coefficients below.
    # The same A also gives the drag relative to the no-slip Stokes drag.
    drag = (1+2*beta)/(1+3*beta)
    A = 3*radius*drag/2
    B = -radius**3/(2*(1+3*beta))

    def integrand(r):
        x = (r-radius)/width
        concentration = np.exp(-strength*(1-x)**2)*solution.sol(x)[0]
        derivative = -2*strength*(1-x)/width
        # r**2 is the direct carried-solute reaction.  The remaining term
        # is the auxiliary-flow contribution from the solution stress.
        weight = r**2 - A*r - B/r
        return weight*concentration*derivative
    return 2/(9*radius*drag)*quad(integrand, radius, end, epsabs=2e-11, epsrel=2e-11)[0]


def predict_at(experiments, viscosity):
    return np.array([response(e['radius'], e['width'], e['strength'], e['slip'])
                     for e in experiments])/viscosity


class Model:
    def __init__(self):
        self.viscosity = None

    def fit(self, records):
        # Every experiment has the form value = response / viscosity.  The
        # weighted least-squares estimate is consequently available in
        # closed form and uses the instrument uncertainties as supplied.
        basis = []
        values = []
        weights = []
        for record in records:
            basis.append(response(**record['input']))
            values.append(float(record['value']))
            sigma = float(record.get('sigma', 1.0))
            if not np.isfinite(sigma) or sigma <= 0:
                raise ValueError('record sigma must be positive and finite')
            weights.append(1.0/(sigma*sigma))

        basis = np.asarray(basis, dtype=float)
        values = np.asarray(values, dtype=float)
        weights = np.asarray(weights, dtype=float)
        denominator = np.sum(weights*basis*basis)
        if not np.isfinite(denominator) or denominator <= 0:
            raise ValueError('calibration records do not identify viscosity')
        numerator = np.sum(weights*basis*values)
        viscosity = denominator/numerator
        if not np.isfinite(viscosity) or viscosity <= 0:
            raise ValueError('calibration records imply invalid viscosity')
        self.viscosity = float(viscosity)
        return self

    def predict(self, experiments):
        return predict_at(experiments, self.viscosity)
