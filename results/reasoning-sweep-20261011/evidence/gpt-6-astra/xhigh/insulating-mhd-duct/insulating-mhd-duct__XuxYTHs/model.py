"""Transient flow with no-slip, electrically insulating duct walls.

The transverse current is written as j = (psi_y, -psi_x).  Insulation
makes psi constant on the connected boundary; choose that constant to be
zero.  Ohm's law and charge conservation then give

    -laplacian(psi) = conductivity * (B . grad) u,
    density * u_t = g + viscosity * laplacian(u) + (B . grad) psi.

Eliminating psi produces a symmetric, dissipative magnetic operator.  A
sine Galerkin basis enforces both boundary conditions.  The electric
inverse Laplacian is evaluated analytically, without truncating its modes.
"""

from functools import lru_cache

import numpy as np
from scipy.linalg import eigh
from scipy.optimize import minimize_scalar
from scipy.special import erfc, roots_legendre

WIDTH, HEIGHT = 0.012, 0.008
DENSITY, CONDUCTIVITY = 1000.0, 1e5
X_MODES, Y_MODES = 64, 48

# Integrate the nonmagnetic heat response with s = time * q**2.  This
# substitution removes the square-root startup singularity in the mean.
_QUAD_POINTS, _QUAD_WEIGHTS = roots_legendre(64)
_QUAD_POINTS = (_QUAD_POINTS + 1) / 2
_QUAD_WEIGHTS = _QUAD_WEIGHTS * _QUAD_POINTS
_QUAD_POINTS = _QUAD_POINTS**2
_HEAT_MODES = np.arange(1, 40, 2)


def _heat_response(tau, position):
    """Dirichlet heat evolution of 1 on a unit interval.

    Use images at short times and a rapidly converging sine series later.
    A position of None requests the spatial mean.
    """
    if position is None:
        result = 1 - 4 * np.sqrt(tau / np.pi)
        weights = 8 / (np.pi * _HEAT_MODES)**2
    else:
        root = 2 * np.sqrt(tau)
        result = 1 - erfc(position / root) - erfc((1 - position) / root)
        weights = 4 * np.sin(np.pi * _HEAT_MODES * position) / (np.pi * _HEAT_MODES)
    late = tau >= 0.01
    result[late] = weights @ np.exp(
        -np.pi**2 * _HEAT_MODES[:, None]**2 * tau[None, late])
    return result


@lru_cache(512)
def _hydrodynamic_response(viscosity, time, x, y):
    """Nonmagnetic velocity per unit pressure acceleration, including startup."""
    if time == 0.0:
        return 0.0
    diffusion_time = viscosity / DENSITY * time * _QUAD_POINTS
    heat = (_heat_response(diffusion_time / WIDTH**2, x)
            * _heat_response(diffusion_time / HEIGHT**2, y))
    return time * float(_QUAD_WEIGHTS @ heat)


def derivative_overlap(target, source):
    """Integral of normalized sine(target) times sine(source)' on [0, 1]."""
    p = np.asarray(target, dtype=float)[:, None]
    m = np.asarray(source, dtype=float)[None, :]
    denominator = p * p - m * m
    return np.divide(
        4 * p * m, denominator, out=np.zeros_like(denominator),
        where=(p + m) % 2 == 1,
    )


def _parallel_block(indices, length, transverse_wave):
    """<d sin_i, (-d^2 + transverse_wave^2)^-1 d sin_j>.

    The Dirichlet Green function gives a diagonal matrix minus one outer
    product within each parity block, including all electric modes.
    """
    k = indices * np.pi / length
    k2 = k * k + transverse_wave * transverse_wave
    boundary = np.tanh(transverse_wave * length / 2)
    if indices[0] % 2:
        boundary = 1 / boundary
    return (np.diag(k * k / k2)
            - (4 * transverse_wave / length * boundary)
            * np.outer(k / k2, k / k2))


@lru_cache(4)
def operators(nx=X_MODES, ny=Y_MODES):
    # Uniform forcing and central inversion symmetry retain odd/odd AND
    # even/even modes.  An oblique field couples these two sets of modes.
    m, n = np.meshgrid(np.arange(1, nx + 1), np.arange(1, ny + 1), indexing='ij')
    keep = (m + n) % 2 == 0
    m, n = m[keep], n[keep]
    wave2 = (m * np.pi / WIDTH)**2 + (n * np.pi / HEIGHT)**2
    mean = np.where(m % 2 == 1, 8 / (np.pi**2 * m * n), 0.0)
    qx = np.zeros((len(m), len(m)))
    qy = np.zeros_like(qx)
    for fixed in np.unique(n):
        ids = np.flatnonzero(n == fixed)
        qx[np.ix_(ids, ids)] = _parallel_block(m[ids], WIDTH, fixed * np.pi / HEIGHT)
    for fixed in np.unique(m):
        ids = np.flatnonzero(m == fixed)
        qy[np.ix_(ids, ids)] = _parallel_block(n[ids], HEIGHT, fixed * np.pi / WIDTH)

    # Dx^* L^-1 Dy + Dy^* L^-1 Dx.  Each term has just one
    # contributing electric mode for any pair of velocity modes.
    dx = derivative_overlap(m, m) / WIDTH
    dy = derivative_overlap(n, n) / HEIGHT
    cross_wave2 = ((m[:, None] * np.pi / WIDTH)**2
                   + (n[None, :] * np.pi / HEIGHT)**2)
    qxy = -dx * dy * (1 / cross_wave2 + 1 / cross_wave2.T)
    return m, n, wave2, mean, qx, qy, qxy


@lru_cache(16)
def spectrum(viscosity, bx, by, nx=X_MODES, ny=Y_MODES):
    m, n, wave2, mean, qx, qy, qxy = operators(nx, ny)
    if bx == 0.0 and by == 0.0:
        ids = np.flatnonzero(m % 2 == 1)
        return [(ids, viscosity / DENSITY * wave2[ids], None, mean[ids])]

    if bx == 0.0:
        groups = [np.flatnonzero((m == fixed) & (n % 2 == 1))
                  for fixed in np.unique(m[m % 2 == 1])]
    elif by == 0.0:
        groups = [np.flatnonzero((n == fixed) & (m % 2 == 1))
                  for fixed in np.unique(n[n % 2 == 1])]
    else:
        groups = [np.arange(len(m))]

    result = []
    for ids in groups:
        ix = np.ix_(ids, ids)
        operator = (CONDUCTIVITY / DENSITY
                    * (bx * bx * qx[ix] + by * by * qy[ix] + bx * by * qxy[ix]))
        operator[np.diag_indices_from(operator)] += viscosity / DENSITY * wave2[ids]
        rates, vectors = eigh(operator, check_finite=False)
        result.append((ids, rates, vectors, vectors.T @ mean[ids]))
    return result


class Model:
    def __init__(self):
        self.viscosity = None

    def fit(self, records):
        """Estimate the common viscosity by Gaussian maximum likelihood."""
        records = list(records)
        if not records:
            raise ValueError('At least one calibration record is required.')
        experiments = [r['input'] for r in records]
        values = np.asarray([r['value'] for r in records], dtype=float)
        errors = np.asarray([r['sigma'] for r in records], dtype=float)
        if not np.all(np.isfinite(values)) or not np.all(np.isfinite(errors) & (errors > 0)):
            raise ValueError('Calibration values must be finite and sigma must be positive.')

        def objective(viscosity):
            residual = (self._predict(experiments, viscosity) - values) / errors
            return float(residual @ residual)

        result = minimize_scalar(objective, bounds=(0.006, 0.018), method='bounded',
                                 options={'xatol': 1e-11})
        # A constrained maximum-likelihood solution can lie on either bound.
        candidates = [float(result.x), 0.006, 0.018]
        self.viscosity = min(candidates, key=objective)
        return self

    def predict(self, experiments):
        """Return mean or point velocities in m/s after a pressure step."""
        if self.viscosity is None:
            raise ValueError('Call fit before predict.')
        return self._predict(experiments, self.viscosity)

    @staticmethod
    def _predict(experiments, viscosity):
        m, n, wave2, mean, *_ = operators()
        viscous_rates = viscosity / DENSITY * wave2
        output = []
        for e in experiments:
            time, gradient = e['time'], e['pressure_gradient']
            if time == 0.0 or gradient == 0.0:
                output.append(0.0)
                continue
            bx, by = e['magnetic_x'], e['magnetic_y']
            if e['observable'] == 'mean':
                observation = mean
                x = y = None
            elif e['observable'] == 'point':
                # A reflection changes the sign of one field component;
                # simultaneous field reversal leaves the flow unchanged.
                x = 1 - e['x'] if bx * by < 0 else e['x']
                y = e['y']
                observation = 2 * np.sin(m * np.pi * x) * np.sin(n * np.pi * y)
            else:
                raise ValueError('observable must be mean or point')
            hydrodynamic = _hydrodynamic_response(viscosity, time, x, y)
            if bx == 0.0 and by == 0.0:
                output.append(gradient / DENSITY * hydrodynamic)
                continue
            value = 0.0
            for ids, rates, vectors, forcing in spectrum(viscosity, abs(bx), abs(by)):
                amplitude = -np.expm1(-rates * time) / rates * forcing
                if vectors is not None:
                    amplitude = vectors @ amplitude
                value += observation[ids] @ amplitude
            # Restore the nonmagnetic high-mode tail.  In particular, this
            # preserves u ~ g*time/density at arbitrarily short times,
            # where a finite sine expansion of uniform forcing is inaccurate.
            truncated_hydrodynamic = observation @ (
                -np.expm1(-viscous_rates * time) / viscous_rates * mean)
            value += hydrodynamic - truncated_hydrodynamic
            output.append(gradient / DENSITY * value)
        return np.asarray(output, dtype=float)
