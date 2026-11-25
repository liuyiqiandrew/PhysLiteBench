from functools import lru_cache
import numpy as np
from scipy.integrate import quad, solve_bvp


@lru_cache(256)
def response(radius, width, strength, slip):
    """Velocity per concentration gradient at unit dynamic viscosity.

    Write the dipolar solute perturbation as exp(-U) f(r) cos(theta).
    Diffusion gives (r^2 exp(-U) f')' = 2 exp(-U) f, with
    f'(a)=0 and f=r+C/r^2 outside the interaction layer.
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
    beta = slip/radius
    drag = (1+2*beta)/(1+3*beta)
    def integrand(r):
        x = (r-radius)/width
        concentration = np.exp(-strength*(1-x)**2)*solution.sol(x)[0]
        derivative = -2*strength*(1-x)/width
        # Reciprocal theorem, including the reaction force on the sphere:
        # the weight is r^2*(1-h), where the radial auxiliary velocity is
        # h = 3*a*drag/(2*r) - a^3/(2*(1+3*beta)*r^3).
        # Factoring in z avoids cancellation in thin interaction layers.
        z = r-radius
        weight = (z*z*(2*r+radius)/(2*r)
                  + 3*radius*beta/(2*(1+3*beta))*z*(r+radius)/r)
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
            raise ValueError('At least one calibration record is required.')
        unit_response = predict_at([r['input'] for r in records], 1.)
        values = np.asarray([r['value'] for r in records], dtype=float)
        sigma = np.asarray([r['sigma'] for r in records], dtype=float)
        if (not np.isfinite(values).all() or not np.isfinite(sigma).all()
                or np.any(sigma <= 0)):
            raise ValueError('Calibration values and positive uncertainties must be finite.')
        # The model is linear in inverse viscosity, so Gaussian maximum
        # likelihood is a weighted one-parameter least-squares fit.
        x, y = unit_response/sigma, values/sigma
        information = np.dot(x, x)
        if information == 0:
            raise ValueError('Calibration does not constrain viscosity.')
        inverse_viscosity = np.clip(np.dot(x, y)/information, 1/1.6, 1/.8)
        self.viscosity = float(1/inverse_viscosity)
        return self

    def predict(self, experiments):
        if self.viscosity is None:
            raise RuntimeError('Fit the model before predicting.')
        return predict_at(experiments, self.viscosity)
