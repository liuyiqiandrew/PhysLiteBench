from functools import lru_cache
import numpy as np
from scipy.integrate import quad, solve_bvp


@lru_cache(256)
def response(radius, width, strength, slip):
    """Return the response with viscosity set to one.

    The concentration perturbation is written as

        c = exp(-U) * (1 + G*f(r)*cos(theta)),

    where ``f`` is matched to the far-field solution ``r + O(r**-2)``.
    The force density exerted by the potential on the solute is then
    transferred with the opposite sign to the fluid.  A reciprocal-theorem
    calculation leaves the carried-potential reaction weighted by
    ``1 - uhat_r/U``; ``uhat`` is the auxiliary flow of a translating sphere.
    This is the part that depends on the Navier traction condition.
    """
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

    slip_ratio = slip/radius
    # Auxiliary flow for a unit translating sphere, in the laboratory frame:
    # uhat_r*cos(theta) = alpha*a/r + beta*a**3/r**3.
    alpha = (3+4*slip_ratio)/(2*(1+3*slip_ratio))
    beta = (-1+2*slip_ratio)/(2*(1+3*slip_ratio))
    drag = (1+2*slip_ratio)/(1+3*slip_ratio)

    def integrand(r):
        x = (r-radius)/width
        concentration = np.exp(-strength*(1-x)**2)*solution.sol(x)[0]
        derivative = -2*strength*(1-x)/width
        auxiliary_radial = alpha*radius/r + beta*radius**3/r**3
        weight = r**2*(1-auxiliary_radial)
        return weight*concentration*derivative

    return 2/(9*radius*drag)*quad(integrand, radius, end, epsabs=2e-11, epsrel=2e-11)[0]


def predict_at(experiments, viscosity):
    return np.asarray([
        response(e['radius'], e['width'], e['strength'], e['slip'])
        for e in experiments
    ], dtype=float)/viscosity


class Model:
    def __init__(self):
        self.viscosity = None

    def fit(self, records):
        # The model is linear in inverse viscosity.  Use the stated Gaussian
        # uncertainties, even though the supplied calibration has a common
        # sigma, so this remains the correct likelihood for the interface.
        basis = np.asarray([
            response(r['input']['radius'], r['input']['width'],
                     r['input']['strength'], r['input']['slip'])
            for r in records
        ], dtype=float)
        values = np.asarray([r['value'] for r in records], dtype=float)
        sigma = np.asarray([r['sigma'] for r in records], dtype=float)
        weights = 1.0/(sigma*sigma)

        inverse_viscosity = np.sum(weights*basis*values)/np.sum(weights*basis*basis)
        self.viscosity = 1.0/inverse_viscosity
        return self

    def predict(self, experiments):
        return predict_at(experiments, self.viscosity)
