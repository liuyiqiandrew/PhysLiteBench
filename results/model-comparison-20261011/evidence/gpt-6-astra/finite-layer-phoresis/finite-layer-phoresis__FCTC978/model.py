from functools import lru_cache
import numpy as np
from scipy.integrate import quad, solve_bvp


@lru_cache(256)
def response(radius, width, strength, slip):
    """Return velocity/gradient at unit viscosity.

    Write the dipolar concentration as G*exp(-U)*h(r)*cos(theta).
    Zero solute flux gives (r**2*exp(-U)*h')' = 2*exp(-U)*h.
    Outside the interaction layer h = r + C/r**2, so the outer
    boundary condition matches the exact unbounded solution.
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
        # For an auxiliary translating Navier-slip sphere,
        # u_hat_r / (V_hat*cos(theta)) = A/r + B/r**3,
        # A = 3*radius*drag/2, B = -radius**3/(2*(1+3*beta)).
        # Force balance includes the reaction of the potential on the
        # sphere. The reciprocal-theorem weight is therefore
        # r**2*(1 - A/r - B/r**3). Factor it to avoid cancellation
        # near the surface, especially for thin interaction layers.
        z = r-radius
        weight = (z**2*(2*r+radius)/(2*r)
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
        """Infer viscosity by bounded Gaussian maximum likelihood."""
        records = list(records)
        if not records:
            raise ValueError('At least one calibration record is required.')
        unit_response = predict_at([r['input'] for r in records], 1.)
        values = np.asarray([r['value'] for r in records], dtype=float)
        sigmas = np.asarray([r['sigma'] for r in records], dtype=float)
        if (not np.isfinite(values).all() or not np.isfinite(sigmas).all()
                or np.any(sigmas <= 0)):
            raise ValueError('Calibration values must be finite and uncertainties positive.')

        # The concentration and unit-viscosity flow do not depend on eta.
        # Thus the response is linear in inverse viscosity, and its
        # weighted least-squares optimum has a closed form.
        design = unit_response/sigmas
        information = np.dot(design, design)
        if information == 0:
            raise ValueError('Calibration must contain a nonzero solute response.')
        inverse_viscosity = np.dot(design, values/sigmas)/information
        inverse_viscosity = np.clip(inverse_viscosity, 1/1.6, 1/.8)
        self.viscosity = float(1/inverse_viscosity)
        return self

    def predict(self, experiments):
        if self.viscosity is None:
            raise RuntimeError('Fit the model before predicting.')
        return predict_at(experiments, self.viscosity)
