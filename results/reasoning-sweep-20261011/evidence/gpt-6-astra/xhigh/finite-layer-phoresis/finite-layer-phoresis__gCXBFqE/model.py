from functools import lru_cache
import numpy as np
from scipy.integrate import quad, solve_bvp


@lru_cache(256)
def _interaction_integrals(radius, width, strength):
    """Radial force integrals, independent of viscosity and surface slip."""
    if strength == 0:
        return 0., 0.

    # Write c = exp(-U) * (c0 + G*f(r)*cos(theta)).  Neglecting
    # advection gives (r**2 * exp(-U) * f')' = 2 * exp(-U) * f.
    # The independent variable x spans the interaction layer, [0, 1].
    end = radius+width

    def equations(x, y):
        r = radius+width*x
        return np.array([y[1], -(2*width/r+2*strength*(1-x))*y[1]+2*width**2*y[0]/r**2])

    def boundary(left, right):
        # No normal flux at the sphere; outside the potential,
        # f(r) = r + C/r**2, which supplies the outer Robin condition.
        return np.array([left[1], end*right[1]/width+2*right[0]-3*end])

    grid = np.linspace(0., 1., 129)
    r = radius+width*grid
    initial = np.array([r+radius**3/(2*r**2), width*(1-radius**3/r**3)])
    solution = solve_bvp(equations, boundary, grid, initial, tol=2e-10, max_nodes=10000)
    if not solution.success:
        raise RuntimeError(solution.message)

    def integrand(x, slip_term):
        z = width*x
        r = radius+z
        concentration = np.exp(-strength*(1-x)**2)*solution.sol(x)[0]
        # U'(r) dr = (dU/dx) dx.
        derivative = -2*strength*(1-x)
        weight = r*z if slip_term else z**2*(2*r+radius)/(2*r)
        return weight*concentration*derivative

    return tuple(quad(integrand, 0., 1., args=(slip_term,),
                      epsabs=2e-11, epsrel=2e-11)[0]
                 for slip_term in (False, True))


def response(radius, width, strength, slip):
    """Signed velocity / concentration gradient at unit viscosity."""
    no_slip_integral, slip_integral = _interaction_integrals(radius, width, strength)
    beta = slip/radius

    # The auxiliary translating Navier sphere has radial velocity
    # u*_r / (V* cos(theta)) = A/r + B/r**3, where
    # A = 3*a*(1+2*beta)/(2*(1+3*beta)),
    # B = -a**3/(2*(1+3*beta)), and drag = 4*pi*viscosity*A.
    # Including the solute's reaction force on the sphere, reciprocity
    # weights the radial force by r**2*(1-A/r-B/r**3).  Factoring this
    # weight avoids cancellation near the surface and gives both terms
    # below.  Slip changes the force weight as well as the drag.
    return 2*(no_slip_integral+3*beta*slip_integral)/(9*radius*(1+2*beta))


def predict_at(experiments, viscosity):
    return np.array([response(e['radius'], e['width'], e['strength'], e['slip'])
                     for e in experiments], dtype=float)/viscosity


class Model:
    def __init__(self):
        self.viscosity = None

    def fit(self, records):
        records = list(records)
        if not records:
            raise ValueError('At least one calibration record is required.')
        unit_response = predict_at([r['input'] for r in records], 1.)
        measured = np.asarray([r['value'] for r in records], dtype=float)
        sigma = np.asarray([r['sigma'] for r in records], dtype=float)
        if not np.all(np.isfinite(measured)) or not np.all(np.isfinite(sigma) & (sigma > 0)):
            raise ValueError('Calibration values must be finite and uncertainties positive.')

        # The forward model is linear in inverse viscosity.  This is the
        # Gaussian maximum-likelihood fit, constrained to the stated range.
        weighted_response = unit_response/sigma
        information = np.dot(weighted_response, weighted_response)
        if information == 0:
            raise ValueError('Calibration records do not constrain viscosity.')
        inverse_viscosity = np.dot(weighted_response, measured/sigma)/information
        self.viscosity = float(1/np.clip(inverse_viscosity, 1/1.6, 1/.8))
        return self

    def predict(self, experiments):
        if self.viscosity is None:
            raise RuntimeError('Fit the model before predicting.')
        return predict_at(experiments, self.viscosity)
