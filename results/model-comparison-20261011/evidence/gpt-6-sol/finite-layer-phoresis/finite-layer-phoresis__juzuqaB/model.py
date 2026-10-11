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
    # Flow around a translating Navier-slip sphere has radial velocity
    # cos(theta) * (flow_stokeslet/r + flow_dipole/r**3).  The reciprocal
    # theorem weights the solute force by one minus this velocity.
    flow_stokeslet = 1.5*radius*drag
    flow_dipole = -radius**3/(2*(1+3*beta))
    def integrand(r):
        x = (r-radius)/width
        concentration = np.exp(-strength*(1-x)**2)*solution.sol(x)[0]
        derivative = -2*strength*(1-x)/width
        weight = r*r-flow_stokeslet*r-flow_dipole/r
        return weight*concentration*derivative
    return 2/(9*radius*drag)*quad(integrand, radius, end, epsabs=2e-11, epsrel=2e-11)[0]


def predict_at(experiments, viscosity):
    return np.array([response(e['radius'], e['width'], e['strength'], e['slip'])
                     for e in experiments])/viscosity


class Model:
    def __init__(self):
        self.viscosity = None

    def fit(self, records):
        records = list(records)
        if not records:
            raise ValueError('At least one calibration record is required')
        unit_responses = predict_at([record['input'] for record in records], 1.)
        measured = np.array([record['value'] for record in records], dtype=float)
        sigma = np.array([record['sigma'] for record in records], dtype=float)
        weighted_response = unit_responses/sigma
        information = np.dot(weighted_response, weighted_response)
        if information == 0:
            raise ValueError('Calibration records do not identify viscosity')
        inverse_viscosity = np.dot(weighted_response, measured/sigma)/information
        inverse_viscosity = np.clip(inverse_viscosity, 1/1.6, 1/.8)
        self.viscosity = float(1/inverse_viscosity)
        return self

    def predict(self, experiments):
        return predict_at(experiments, self.viscosity)
