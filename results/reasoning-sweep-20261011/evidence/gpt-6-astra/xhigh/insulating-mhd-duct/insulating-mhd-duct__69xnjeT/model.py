"""Transient, fully developed MHD flow in an insulating rectangular duct.

Write the cross-sectional current as j = (psi_y, -psi_x). Charge
conservation and insulating walls then give

    -laplacian(psi) = sigma * (Bx*u_x + By*u_y),   psi|wall = 0,
    rho*u_t = g + eta*laplacian(u) + Bx*psi_x + By*psi_y.

Eliminating psi gives a positive magnetic damping operator. A sine
Galerkin discretization enforces both wall conditions and permits exact
time evolution of the resulting linear system from rest.
"""

from functools import lru_cache

import numpy as np
from scipy.linalg import eigh
from scipy.optimize import minimize_scalar
from scipy.special import erfc

WIDTH, HEIGHT = 0.012, 0.008
DENSITY, CONDUCTIVITY = 1000.0, 1e5
X_MODES, Y_MODES = 64, 48


# Integrating over sqrt(time) removes the initial square-root singularity
# in the mean heat content. This also resolves arbitrarily short startups.
_quad_x, _quad_w = np.polynomial.legendre.leggauss(64)
_quad_x = (_quad_x + 1) / 2
_quad_w = _quad_w * _quad_x


def _heat_content(s, coordinate):
    """Diffusion of an initially uniform unit field on a unit interval."""
    result = np.empty_like(s)
    short = s < 0.01
    if coordinate is None:
        result[short] = 1 - 4 * np.sqrt(s[short] / np.pi)
    else:
        scale = 2 * np.sqrt(s[short])
        result[short] = 1 - erfc(coordinate / scale) - erfc((1 - coordinate) / scale)
    # At small s the omitted image terms are below 1e-12. At larger s,
    # this complementary sine series converges exponentially quickly.
    odd = np.arange(1, 50, 2)
    weights = (8 / (np.pi * odd)**2 if coordinate is None else
               4 * np.sin(np.pi * odd * coordinate) / (np.pi * odd))
    result[~short] = weights @ np.exp(-np.pi**2 * odd[:, None]**2 * s[None, ~short])
    return result


@lru_cache(512)
def _hydrodynamic_response(viscosity, time, x, y):
    """Unmagnetized velocity per unit pressure acceleration, in seconds."""
    diffusion_time = viscosity / DENSITY * time * _quad_x**2
    hx = _heat_content(diffusion_time / WIDTH**2, x)
    hy = _heat_content(diffusion_time / HEIGHT**2, y)
    return time * (_quad_w @ (hx * hy))


def derivative_overlap(target, source):
    """Area-normalized sine/derivative overlap, before division by length."""
    p = np.asarray(target, dtype=float)[:, None]
    m = np.asarray(source, dtype=float)[None, :]
    return np.divide(
        4 * p * m, p * p - m * m,
        out=np.zeros((len(p), m.shape[1])), where=(p + m) % 2 == 1,
    )


def _directional_block(modes, length, transverse_wave):
    """Exact <d(sine), (-laplacian_D)^-1 d(sine)> for one transverse mode.

    The one-dimensional Dirichlet Green function sums all electric
    modes analytically; there is no separate electric-mode truncation.
    All entries in modes have the same parity.
    """
    k = modes * np.pi / length
    a = transverse_wave
    denominator = k * k + a * a
    factor = np.tanh(a * length / 2)
    if modes[0] % 2:
        factor = 1 / factor
    v = k / denominator
    return np.diag(k * k / denominator) - (4 * a / length * factor) * np.outer(v, v)


@lru_cache(4)
def operators(nx=X_MODES, ny=Y_MODES):
    m, n = np.meshgrid(np.arange(1, nx + 1), np.arange(1, ny + 1), indexing='ij')
    # Central inversion symmetry holds even for oblique fields. Separate
    # x/y reflection symmetry does not: even/even modes must be retained.
    keep = (m + n) % 2 == 0
    m, n = m[keep], n[keep]
    wave2 = (m * np.pi / WIDTH)**2 + (n * np.pi / HEIGHT)**2
    mean = np.where(m % 2 == 1, 8 / (np.pi**2 * m * n), 0.0)
    qx = np.zeros((len(m), len(m)))
    qy = np.zeros_like(qx)
    for fixed in np.unique(n):
        ids = np.flatnonzero(n == fixed)
        qx[np.ix_(ids, ids)] = _directional_block(m[ids], WIDTH, fixed * np.pi / HEIGHT)
    for fixed in np.unique(m):
        ids = np.flatnonzero(m == fixed)
        qy[np.ix_(ids, ids)] = _directional_block(n[ids], HEIGHT, fixed * np.pi / WIDTH)

    # D_x.T K^-1 D_y + D_y.T K^-1 D_x. Only opposite parities
    # couple, and the two Kronecker deltas leave no infinite sum.
    mixed_wave2 = (m[:, None] * np.pi / WIDTH)**2 + (n[None, :] * np.pi / HEIGHT)**2
    qxy = -derivative_overlap(m, m) * derivative_overlap(n, n) / (WIDTH * HEIGHT)
    qxy *= 1 / mixed_wave2 + 1 / mixed_wave2.T
    return m, n, wave2, mean, qx, qy, qxy


@lru_cache(16)
def spectrum(viscosity, bx, by, nx=X_MODES, ny=Y_MODES):
    m, n, wave2, mean, qx, qy, qxy = operators(nx, ny)
    if bx == 0 and by == 0:
        ids = np.flatnonzero(m % 2 == 1)
        return [(ids, viscosity / DENSITY * wave2[ids], None, mean[ids])]

    operator = CONDUCTIVITY / DENSITY * (bx * bx * qx + by * by * qy + bx * by * qxy)
    operator[np.diag_indices_from(operator)] += viscosity / DENSITY * wave2
    if bx == 0:
        groups = [np.flatnonzero((m == fixed) & (n % 2 == 1))
                  for fixed in np.unique(m[m % 2 == 1])]
    elif by == 0:
        groups = [np.flatnonzero((n == fixed) & (m % 2 == 1))
                  for fixed in np.unique(n[n % 2 == 1])]
    else:
        groups = [np.arange(len(m))]
    result = []
    for ids in groups:
        rates, vectors = eigh(operator[np.ix_(ids, ids)], check_finite=False)
        result.append((ids, rates, vectors, vectors.T @ mean[ids]))
    return result


def _predict(viscosity, experiments, nx=X_MODES, ny=Y_MODES):
    m, n, wave2, mean, *_ = operators(nx, ny)
    viscous_rates = viscosity / DENSITY * wave2
    output = np.zeros(len(experiments), dtype=float)
    for index, e in enumerate(experiments):
        if e['time'] == 0 or e['pressure_gradient'] == 0:
            continue
        x, y = (None, None) if e['observable'] == 'mean' else (e['x'], e['y'])
        base = _hydrodynamic_response(viscosity, e['time'], x, y)
        if e['magnetic_x'] == 0 and e['magnetic_y'] == 0:
            output[index] = e['pressure_gradient'] / DENSITY * base
            continue
        observation = (mean if e['observable'] == 'mean' else
                       2 * np.sin(m * np.pi * e['x']) * np.sin(n * np.pi * e['y']))
        value = 0.0
        for ids, rates, vectors, forcing in spectrum(
                viscosity, e['magnetic_x'], e['magnetic_y'], nx, ny):
            amplitude = -np.expm1(-rates * e['time']) / rates * forcing
            if vectors is not None:
                amplitude = vectors @ amplitude
            value += observation[ids] @ amplitude
        # Restore the unmagnetized high-mode tail. It carries the singular
        # initial boundary layer; the magnetic correction is much smoother.
        truncated_base = observation @ (
            -np.expm1(-viscous_rates * e['time']) / viscous_rates * mean)
        value += base - truncated_base
        output[index] = e['pressure_gradient'] / DENSITY * value
    return output


class Model:
    def __init__(self):
        self.viscosity = None

    def fit(self, records):
        """Fit the common viscosity by Gaussian weighted least squares."""
        records = list(records)
        if not records:
            raise ValueError('At least one calibration record is required.')
        experiments = [r['input'] for r in records]
        measured = np.asarray([r['value'] for r in records], dtype=float)
        uncertainty = np.asarray([r['sigma'] for r in records], dtype=float)
        if not np.all(np.isfinite(measured)) or not np.all(np.isfinite(uncertainty) & (uncertainty > 0)):
            raise ValueError('Calibration values must be finite and sigma must be positive and finite.')

        def objective(viscosity):
            residual = (_predict(viscosity, experiments) - measured) / uncertainty
            return float(residual @ residual)

        optimum = minimize_scalar(objective, bounds=(0.006, 0.018), method='bounded',
                                  options={'xatol': 1e-10})
        # A constrained maximum-likelihood estimate can lie on a bound.
        self.viscosity = float(min((0.006, optimum.x, 0.018), key=objective))
        return self

    def predict(self, experiments):
        """Return mean or point velocities in m/s, in experiment order."""
        if self.viscosity is None:
            raise ValueError('Fit the model before predicting.')
        return _predict(self.viscosity, list(experiments))
