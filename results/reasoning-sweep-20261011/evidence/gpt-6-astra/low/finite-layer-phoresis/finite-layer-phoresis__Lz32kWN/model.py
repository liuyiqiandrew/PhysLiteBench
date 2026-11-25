from functools import lru_cache
import numpy as np
from scipy.integrate import quad, solve_bvp


@lru_cache(256)
def response(radius, width, strength, slip):
    """Velocity/gradient at unit viscosity, including finite interaction width."""
    if strength == 0:
        return 0.
    end = radius+width
    # Write the dipolar concentration as G exp(-U) h(r) cos(theta).
    # Then (r^2 exp(-U) h')' = 2 exp(-U) h. Outside the
    # interaction layer h = r + B/r^2; eliminate B at the outer edge.
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
        # Reciprocal theorem with a translating Navier-slip sphere:
        # weight = r^2 (1 - u_aux,r / cos(theta)). The 1 includes
        # the reaction of the solute force on the freely moving sphere.
        # This factored form avoids cancellation close to the surface.
        weight = ((r-radius)**2*(2*r+radius)/(2*r)
                  + 3*radius*beta/(2*(1+3*beta))
                  * (r-radius)*(r+radius)/r)
        return weight*concentration*derivative
    return 2/(9*radius*drag)*quad(integrand, radius, end, epsabs=2e-11, epsrel=2e-11)[0]


def predict_at(experiments, viscosity):
    return np.array([response(e['radius'], e['width'], e['strength'], e['slip'])
                     for e in experiments])/viscosity


class Model:
    def __init__(self):
        self.viscosity = None

    def fit(self, records):
        """Infer viscosity using the known independent Gaussian errors."""
        records = list(records)
        if not records:
            raise ValueError('Calibration records must not be empty.')
        unit = predict_at([r['input'] for r in records], 1.)
        values = np.asarray([r['value'] for r in records], dtype=float)
        sigma = np.asarray([r['sigma'] for r in records], dtype=float)
        if (not np.isfinite(values).all() or not np.isfinite(sigma).all()
                or np.any(sigma <= 0)):
            raise ValueError('Calibration values and positive errors must be finite.')
        design = unit/sigma
        information = np.dot(design, design)
        if information == 0:
            raise ValueError('Calibration does not identify viscosity.')
        # The response is linear in inverse viscosity, so the constrained
        # maximum-likelihood fit has a closed form.
        inverse_viscosity = np.dot(design, values/sigma)/information
        self.viscosity = float(1/np.clip(inverse_viscosity, 1/1.6, 1/.8))
        return self

    def predict(self, experiments):
        if self.viscosity is None:
            raise RuntimeError('Fit the model before predicting.')
        return predict_at(experiments, self.viscosity)
