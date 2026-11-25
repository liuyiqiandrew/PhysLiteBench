"""Transient, fully developed magnetohydrodynamic duct flow.

The velocity and electric-current stream function are expanded in sine
functions. Both vanish at the wall: the first condition is no slip and the
second is the insulating-wall condition. Eliminating the stream function
gives a symmetric positive magnetic damping operator for the velocity.
"""

from functools import lru_cache

import numpy as np
from scipy.linalg import eigh
from scipy.optimize import minimize_scalar


WIDTH, HEIGHT = 0.012, 0.008
DENSITY, CONDUCTIVITY = 1000.0, 1.0e5
X_MODES, Y_MODES = 48, 64


def derivative_overlap(target, source):
    """Integral of a normalized sine mode times a sine mode's derivative."""
    p = np.asarray(target, dtype=float)[:, None]
    m = np.asarray(source, dtype=float)[None, :]
    denominator = p * p - m * m
    return np.divide(
        4 * p * m,
        denominator,
        out=np.zeros_like(denominator),
        where=(np.remainder(p + m, 2) == 1),
    )


@lru_cache(maxsize=2)
def operators(nx=X_MODES, ny=Y_MODES):
    m, n = np.meshgrid(np.arange(1, nx + 1), np.arange(1, ny + 1), indexing="ij")
    # Constant pressure forcing occupies odd/odd modes. Magnetic coupling
    # preserves the parity of m+n and also excites even/even modes.
    keep = (m + n) % 2 == 0
    m, n = m[keep], n[keep]
    wave2 = (m * np.pi / WIDTH) ** 2 + (n * np.pi / HEIGHT) ** 2
    forcing = np.where(m % 2 == 1, 8 / (np.pi**2 * m * n), 0.0)
    size = len(m)
    qx = np.zeros((size, size))
    qy = np.zeros((size, size))

    # The current stream function has Dirichlet wall data. Resolve its
    # inverse Laplacian with a longer auxiliary sine series.
    electric = np.arange(1, 401)
    for fixed in np.unique(n):
        ids = np.flatnonzero(n == fixed)
        d = derivative_overlap(electric, m[ids]) / WIDTH
        k2 = (electric * np.pi / WIDTH) ** 2 + (fixed * np.pi / HEIGHT) ** 2
        qx[np.ix_(ids, ids)] = d.T @ (d / k2[:, None])
    for fixed in np.unique(m):
        ids = np.flatnonzero(m == fixed)
        d = derivative_overlap(electric, n[ids]) / HEIGHT
        k2 = (fixed * np.pi / WIDTH) ** 2 + (electric * np.pi / HEIGHT) ** 2
        qy[np.ix_(ids, ids)] = d.T @ (d / k2[:, None])

    # Cross term Dx.T (-Laplacian)^-1 Dy plus its transpose. Each pair has
    # exactly one shared auxiliary mode, so it can be evaluated directly.
    dm = derivative_overlap(m, m) / WIDTH
    dn = derivative_overlap(n, n) / HEIGHT
    k2 = (m[None, :] * np.pi / WIDTH) ** 2 + (n[:, None] * np.pi / HEIGHT) ** 2
    mixed = dm.T * dn / k2
    qxy = mixed + mixed.T
    return m, n, wave2, forcing, qx, qy, qxy


@lru_cache(maxsize=8)
def spectrum(viscosity, bx, by, nx=X_MODES, ny=Y_MODES):
    m, n, wave2, forcing, qx, qy, qxy = operators(nx, ny)
    magnetic = bx * bx * qx + bx * by * qxy + by * by * qy
    matrix = np.diag(viscosity / DENSITY * wave2)
    matrix += CONDUCTIVITY / DENSITY * magnetic
    if bx == 0.0 and by == 0.0:
        groups = [np.flatnonzero(forcing != 0)]
    elif bx == 0.0:
        groups = [np.flatnonzero((m == fixed) & (n % 2 == 1))
                  for fixed in np.unique(m[m % 2 == 1])]
    elif by == 0.0:
        groups = [np.flatnonzero((n == fixed) & (m % 2 == 1))
                  for fixed in np.unique(n[n % 2 == 1])]
    else:
        groups = [np.arange(len(m))]
    result = []
    for ids in groups:
        if bx == 0.0 and by == 0.0:
            rates = matrix.diagonal()[ids]
            vectors = np.eye(len(ids))
        else:
            rates, vectors = eigh(matrix[np.ix_(ids, ids)], check_finite=False)
        result.append((ids, rates, vectors, vectors.T @ forcing[ids]))
    return result


class Model:
    def __init__(self):
        self.viscosity = None

    def fit(self, records):
        records = list(records)
        if not records:
            raise ValueError("At least one calibration record is required")
        experiments = [r["input"] for r in records]
        measured = np.asarray([r["value"] for r in records], dtype=float)
        sigma = np.asarray([r["sigma"] for r in records], dtype=float)
        if np.any(sigma <= 0) or not np.all(np.isfinite(sigma)):
            raise ValueError("Measurement sigma must be positive and finite")

        def objective(viscosity):
            self.viscosity = viscosity
            residual = (self.predict(experiments) - measured) / sigma
            return float(residual @ residual)

        result = minimize_scalar(
            objective, bounds=(0.006, 0.018), method="bounded",
            options={"xatol": 1e-8},
        )
        self.viscosity = float(result.x)
        return self

    def predict(self, experiments):
        if self.viscosity is None:
            raise ValueError("Fit the model before prediction")
        experiments = list(experiments)
        m, n, _, mean, *_ = operators()
        output = np.zeros(len(experiments))
        for i, experiment in enumerate(experiments):
            gradient = float(experiment["pressure_gradient"])
            time = float(experiment["time"])
            if gradient == 0.0 or time == 0.0:
                continue
            bx = float(experiment["magnetic_x"])
            by = float(experiment["magnetic_y"])
            if experiment["observable"] == "mean":
                observation = mean
            else:
                observation = (2 * np.sin(m * np.pi * experiment["x"])
                               * np.sin(n * np.pi * experiment["y"]))
            value = 0.0
            for ids, rates, vectors, forcing in spectrum(self.viscosity, bx, by):
                response = -np.expm1(-rates * time) / rates
                value += observation[ids] @ (vectors @ (response * forcing))
            output[i] = gradient / DENSITY * value
        return output
