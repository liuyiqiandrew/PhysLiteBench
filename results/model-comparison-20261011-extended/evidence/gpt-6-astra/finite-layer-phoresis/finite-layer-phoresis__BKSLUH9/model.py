from functools import lru_cache
import numpy as np
from scipy.integrate import quad, solve_bvp


@lru_cache(256)
def response(radius, width, strength, slip):
    """Velocity per concentration gradient at unit viscosity.

    Write the dipolar concentration as G*exp(-U)*f(r)*cos(theta).
    No solute flux gives f'(radius)=0; matching to r+C/r**2 outside
    the potential gives end*f'(end)+2*f(end)=3*end.
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
        # The auxiliary translating Navier-slip sphere has radial velocity
        # H(r)*cos(theta), H = 3*radius*drag/(2*r)
        #                         - radius**3/(2*(1+3*beta)*r**3).
        # Including the reaction force on the particle, the reciprocal
        # theorem weights the radial solute force by r**2*(1-H).
        # Factor at r=radius to avoid subtracting nearly equal terms.
        z = r-radius
        weight = z*(z*(2*r+radius)+6*beta*r**2)/(2*r*(1+3*beta))
        return weight*concentration*derivative
    return 2/(9*radius*drag)*quad(integrand, radius, end, epsabs=2e-11, epsrel=2e-11)[0]


def predict_at(experiments, viscosity):
    return np.array([response(e['radius'], e['width'], e['strength'], e['slip'])
                     for e in experiments], dtype=float)/viscosity


class Model:
    def __init__(self):
        self.viscosity = None

    def fit(self, records):
        """Fit the inverse viscosity, the linear scale of every response."""
        records = list(records)
        if not records:
            raise ValueError('At least one calibration record is required.')
        unit_response = predict_at([r['input'] for r in records], 1.)
        values = np.asarray([r['value'] for r in records], dtype=float)
        sigma = np.asarray([r['sigma'] for r in records], dtype=float)
        if (not np.isfinite(values).all() or not np.isfinite(sigma).all()
                or np.any(sigma <= 0)):
            raise ValueError('Calibration values and positive uncertainties must be finite.')
        design = unit_response/sigma
        information = np.dot(design, design)
        if information == 0:
            raise ValueError('Calibration records do not constrain viscosity.')
        inverse_viscosity = np.dot(design, values/sigma)/information
        # The specified physical range is a constraint on the Gaussian MLE.
        inverse_viscosity = np.clip(inverse_viscosity, 1/1.6, 1/0.8)
        self.viscosity = float(1/inverse_viscosity)
        return self

    def predict(self, experiments):
        if self.viscosity is None:
            raise RuntimeError('Fit the model before making predictions.')
        return predict_at(experiments, self.viscosity)
