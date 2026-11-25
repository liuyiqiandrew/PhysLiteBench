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
        raise NotImplementedError('Implement calibration fitting.')

    def predict(self, experiments):
        return predict_at(experiments, self.viscosity)
