"""Transient conducting flow in the insulating rectangular duct."""

from functools import lru_cache

import numpy as np
from scipy.linalg import eigh
from scipy.optimize import minimize_scalar
from scipy.sparse import coo_matrix


WIDTH, HEIGHT = .012, .008
DENSITY, CONDUCTIVITY = 1000., 1e5
X_MODES, Y_MODES = 40, 56
ELECTRIC_MODES = 400


def _cos_sine_overlap(cos_mode, sine_mode):
    """Average of a normalized cosine and an unnormalised sine."""
    c, s = int(cos_mode), int(sine_mode)
    if c == 0:
        return (1. - (-1.) ** s) / (np.pi * s)
    if (c + s) % 2 == 0:
        return 0.
    return np.sqrt(2.) * 2. * s / (np.pi * (s*s-c*c))


def derivative_overlap(target, source):
    """Integral of normalized sine(target) times derivative of sine(source)."""
    p, m = np.asarray(target, float)[:, None], np.asarray(source, float)[None, :]
    denominator = p*p-m*m
    return np.divide(4*p*m, denominator, out=np.zeros_like(denominator),
                     where=((p+m) % 2 == 1))


@lru_cache(4)
def operators(nx=X_MODES, ny=Y_MODES):
    """Build the flow modes and the complete oblique-field Lorentz operator.

    The flow basis is ``2 sin(m pi x/W) sin(n pi y/H)``.  The transverse
    current is the solenoidal projection of ``u x B``.  Its projection is
    computed in a cosine-cosine Neumann basis, so both motional-current terms
    use one potential and one total current.
    """
    mm, nn = np.meshgrid(np.arange(1, nx + 1), np.arange(1, ny + 1),
                         indexing='ij')
    keep = (mm + nn) % 2 == 0
    m, n = mm[keep], nn[keep]
    count = len(m)
    alpha = m * np.pi / WIDTH
    beta = n * np.pi / HEIGHT
    wave2 = alpha*alpha + beta*beta
    mean = np.where(m % 2 == 1, 8. / (np.pi*np.pi*m*n), 0.)

    # Only potential modes (k=m,l arbitrary) and (k arbitrary,l=n) can
    # overlap derivatives of the flow modes.
    potentials = set()
    for mi in np.unique(m):
        for ell in range(ELECTRIC_MODES + 1):
            potentials.add((int(mi), ell))
    for ni in np.unique(n):
        for kk in range(ELECTRIC_MODES + 1):
            potentials.add((kk, int(ni)))

    flow_m = set(int(v) for v in np.unique(m))
    flow_n = set(int(v) for v in np.unique(n))
    dx_r, dx_c, dx_v = [], [], []
    dy_r, dy_c, dy_v = [], [], []
    potential_k2 = []
    row_number = 0

    for k, ell in sorted(potentials):
        dx_row, dy_row = [], []
        if k in flow_m:
            for j, (mj, nj) in enumerate(zip(m, n)):
                if int(mj) == k:
                    overlap = _cos_sine_overlap(ell, int(nj))
                    if overlap:
                        dx_row.append((j, np.sqrt(2.) * alpha[j] * overlap))
        if ell in flow_n:
            for j, (mj, nj) in enumerate(zip(m, n)):
                if int(nj) == ell:
                    overlap = _cos_sine_overlap(k, int(mj))
                    if overlap:
                        dy_row.append((j, np.sqrt(2.) * beta[j] * overlap))
        if not dx_row and not dy_row:
            continue
        for j, value in dx_row:
            dx_r.append(row_number)
            dx_c.append(j)
            dx_v.append(value)
        for j, value in dy_row:
            dy_r.append(row_number)
            dy_c.append(j)
            dy_v.append(value)
        potential_k2.append((k*np.pi/WIDTH)**2 + (ell*np.pi/HEIGHT)**2)
        row_number += 1

    shape = (row_number, count)
    dx = coo_matrix((dx_v, (dx_r, dx_c)), shape=shape).tocsr()
    dy = coo_matrix((dy_v, (dy_r, dy_c)), shape=shape).tocsr()
    inv_k2 = 1. / np.asarray(potential_k2)

    dx_k_dx = dx.T @ dx.multiply(inv_k2[:, None])
    dy_k_dy = dy.T @ dy.multiply(inv_k2[:, None])
    dy_k_dx = dy.T @ dx.multiply(inv_k2[:, None])

    # Q is defined by <u x B, P(u x B)> and hence is positive semidefinite.
    # Bx uses the y component of u x B; By uses its x component.
    qx = np.eye(count) - dy_k_dy.toarray()
    qy = np.eye(count) - dx_k_dx.toarray()
    qxy = (dy_k_dx + dy_k_dx.T).toarray()
    qx = (qx + qx.T) * .5
    qy = (qy + qy.T) * .5
    qxy = (qxy + qxy.T) * .5
    return m, n, wave2, mean, qx, qy, qxy


@lru_cache(64)
def spectrum(viscosity, bx, by, nx=X_MODES, ny=Y_MODES):
    m, n, wave2, mean, qx, qy, qxy = operators(nx, ny)
    magnetic = bx*bx*qx + by*by*qy + bx*by*qxy
    operator = np.diag(viscosity/DENSITY*wave2) + CONDUCTIVITY/DENSITY*magnetic
    rates, vectors = eigh(operator, check_finite=False)
    return rates, vectors, vectors.T @ mean


def _response(viscosity, experiment):
    m, n, _, mean, *_ = operators()
    bx = float(experiment['magnetic_x'])
    by = float(experiment['magnetic_y'])
    rates, vectors, forcing = spectrum(float(viscosity), bx, by)
    t = float(experiment['time'])
    integrated = np.where(rates > 1e-14,
                          -np.expm1(-rates*t) / rates,
                          t)
    amplitudes = vectors @ (integrated * forcing)
    if experiment['observable'] == 'mean':
        observation = mean
    else:
        observation = 2. * np.sin(m*np.pi*float(experiment['x'])) \
            * np.sin(n*np.pi*float(experiment['y']))
    return float(experiment['pressure_gradient']) / DENSITY \
        * (observation @ amplitudes)


class Model:
    def __init__(self):
        self.viscosity = None

    def fit(self, records):
        """Fit viscosity using the independent Gaussian measurements."""
        inputs = [record['input'] for record in records]
        values = np.asarray([record['value'] for record in records], dtype=float)
        sigmas = np.asarray([record['sigma'] for record in records], dtype=float)

        def objective(viscosity):
            predictions = np.asarray([_response(viscosity, e) for e in inputs])
            residual = (predictions - values) / sigmas
            return float(residual @ residual)

        result = minimize_scalar(objective, bounds=(.006, .018), method='bounded',
                                 options={'xatol': 2e-9, 'maxiter': 40})
        self.viscosity = float(result.x)
        return self

    def predict(self, experiments):
        if self.viscosity is None:
            raise RuntimeError('fit must be called before predict')
        return np.asarray([_response(self.viscosity, e) for e in experiments],
                          dtype=float)
