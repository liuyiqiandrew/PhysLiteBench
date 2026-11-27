"""Count states by radial quadrature in physical radius. Invert the initial
radial action numerically, using the same angular momentum in both fields."""
import numpy as np
from numpy.polynomial.legendre import leggauss

TRUE_PARAMETER = 2.73


def potential(r, field):
    m, b = field
    return -m / (b + np.sqrt(r*r + b*b))


def turning_radii(e, angular, field):
    m, b = field
    k2 = -2 * e
    axis = m / k2
    span = np.sqrt(np.maximum((axis - b)**2 - angular**2 / k2, 0))
    sa = axis + span
    small = angular**2 / (k2 * (sa - b))
    rp = np.sqrt(small * (2*b + small))
    ra = np.sqrt((sa - b) * (sa + b))
    return rp, ra


def radial_measure(e, angular, field, grid):
    theta, weights = grid
    rp, ra = turning_radii(e, angular, field)
    rp, ra = rp[..., None], ra[..., None]
    r = rp + (ra-rp) * np.sin(theta/2)**2
    b = field[1]
    s = np.hypot(r, b)
    # dr/v_r after cancelling the integrable turning-point singularities.
    dt = r / np.sqrt(-2*e)[..., None] * np.sqrt(
        (s + np.hypot(rp, b)) * (s + np.hypot(ra, b)) / ((r+rp)*(ra+r)))
    return r, dt * weights, (ra-rp) * np.sin(theta) / 2


def action(e, angular, field, grid):
    _, time, dr = radial_measure(e, angular, field, grid)
    return np.sum((dr * grid[1])**2 / time, axis=-1) / np.pi


def signal(q, exponent=TRUE_PARAMETER, order=48):
    nodes, weights = leggauss(order)
    grid = ((nodes+1) * np.pi/2, weights * np.pi/2)
    fields = [q["slow"], *q["steps"]]
    field = fields[-1]
    rp, ra = q["peri"], q["apo"]
    m, b = field
    sp, sa = np.hypot([rp, ra], b)
    # Equate Phi+L^2/(2r^2) at the two turning points.
    angular = np.sqrt(2*m*rp**2*ra**2 / ((sp+b)*(sa+b)*(sp+sa)))
    e = np.asarray(potential(rp, field) + angular**2 / (2*rp**2))
    _, time, _ = radial_measure(e, angular, field, grid)
    volume = 16*np.pi**2 * angular * np.sum(time)

    def density(e, index):
        bound = e < 0
        # Use the circular orbit for unused unbound entries.
        mass, scale = fields[index]
        circular = (angular + np.hypot(angular, 2*np.sqrt(mass*scale))) / 2
        safe = np.where(bound, e, -0.5*(mass/circular)**2)
        if index == 0:
            target = action(safe, angular, fields[0], grid)
            start = (angular + np.hypot(angular, 2)) / 2
            momenta = np.geomspace(start, start + 1.1*np.max(target) + 1, 160)
            energy_grid = -0.5 / momenta**2
            actions = action(energy_grid, angular, (1, 1), grid)
            binding = 0.5 / np.interp(target, actions, momenta)**2
            return np.where(bound, (binding / 0.4)**exponent, 0)
        r, time, _ = radial_measure(safe, angular, fields[index], grid)
        old_e = safe[..., None] + potential(r, fields[index-1]) - potential(r, fields[index])
        past = density(old_e, index-1)
        return np.where(bound, np.sum(past*time, axis=-1) / np.sum(time, axis=-1), 0)

    return float(volume * density(e, len(fields)-1))
