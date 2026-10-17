"""Transient Hartmann flow in an electrically insulated rectangular duct.

The velocity is expanded in no-slip sine modes.  The electric potential is
eliminated analytically: for each velocity field it is the Neumann solution
which makes the total motional current divergence-free and tangent to walls.
"""

from functools import lru_cache

import numpy as np
from scipy.linalg import eigh
from scipy.optimize import minimize_scalar


WIDTH, HEIGHT = 0.012, 0.008
DENSITY, CONDUCTIVITY = 1000.0, 1.0e5
X_MODES, Y_MODES = 40, 56


def cosine_sine_overlap(cosine, sine):
    """Integral of cos(cosine*pi*x) sin(sine*pi*x) on [0, 1]."""
    p, m = np.asarray(cosine, float), np.asarray(sine, float)
    plus, minus = m + p, m - p
    first = (1.0 - (-1.0) ** plus) / (np.pi * plus)
    second = np.divide(1.0 - (-1.0) ** minus, np.pi * minus,
                       out=np.zeros(np.broadcast(p, m).shape), where=minus != 0)
    return 0.5 * (first + second)


@lru_cache(maxsize=4)
def operators(nx=X_MODES, ny=Y_MODES):
    """Modal viscous, observation, and single-current magnetic operators."""
    m, n = np.meshgrid(np.arange(1, nx + 1), np.arange(1, ny + 1), indexing="ij")
    keep = (m + n) % 2 == 0
    m, n = m[keep], n[keep]
    count = len(m)
    wave2 = (m * np.pi / WIDTH) ** 2 + (n * np.pi / HEIGHT) ** 2
    mean = np.where((m % 2 == 1) & (n % 2 == 1), 8.0 / (np.pi ** 2 * m * n), 0.0)

    # Bx du/dy and By du/dx potential projections.  The finite cosine sum
    # enforces both insulating-wall conditions; its tail is negligible here.
    electric = np.arange(0, 1201)
    qx = np.zeros((count, count))
    qy = np.zeros((count, count))
    for fixed in np.unique(n):
        ids = np.flatnonzero(n == fixed)
        p = electric[:, None]
        source = (fixed * np.pi / HEIGHT * np.sqrt(2.0 * (2.0 - (p == 0)))
                  * cosine_sine_overlap(p, m[ids][None, :]))
        laplace = (electric * np.pi / WIDTH) ** 2 + (fixed * np.pi / HEIGHT) ** 2
        qx[np.ix_(ids, ids)] = np.eye(len(ids)) - source.T @ (source / laplace[:, None])
    for fixed in np.unique(m):
        ids = np.flatnonzero(m == fixed)
        p = electric[:, None]
        source = (fixed * np.pi / WIDTH * np.sqrt(2.0 * (2.0 - (p == 0)))
                  * cosine_sine_overlap(p, n[ids][None, :]))
        laplace = (fixed * np.pi / WIDTH) ** 2 + (electric * np.pi / HEIGHT) ** 2
        qy[np.ix_(ids, ids)] = np.eye(len(ids)) - source.T @ (source / laplace[:, None])

    # Both source terms share potential mode (M,n).  This is the term that
    # independent axis damping operators miss for an oblique field.
    m_row, n_row = m[:, None], n[:, None]
    m_col, n_col = m[None, :], n[None, :]
    laplace = (m_col * np.pi / WIDTH) ** 2 + (n_row * np.pi / HEIGHT) ** 2
    one_way = (4.0 * np.pi ** 2 * n_row * m_col / (HEIGHT * WIDTH)
               * cosine_sine_overlap(m_col, m_row)
               * cosine_sine_overlap(n_row, n_col) / laplace)
    qxy = one_way + one_way.T
    return m, n, wave2, mean, qx, qy, qxy


@lru_cache(maxsize=64)
def spectrum(viscosity, bx, by, nx=X_MODES, ny=Y_MODES):
    m, n, wave2, mean, qx, qy, qxy = operators(nx, ny)
    magnetic = bx * bx * qx + by * by * qy + bx * by * qxy
    operator = np.diag(viscosity / DENSITY * wave2) + CONDUCTIVITY / DENSITY * magnetic
    # An oblique field couples odd/odd modes to even/even modes through qxy.
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
        values, vectors = eigh(operator[np.ix_(ids, ids)], check_finite=False)
        result.append((ids, values, vectors, vectors.T @ mean[ids]))
    return result


class Model:
    def __init__(self):
        self.viscosity = None

    def fit(self, records):
        records = list(records)
        if not records:
            raise ValueError("at least one calibration record is required")
        experiments = [record["input"] for record in records]
        measured = np.asarray([record["value"] for record in records], dtype=float)
        uncertainty = np.asarray([record["sigma"] for record in records], dtype=float)
        if not (np.isfinite(measured).all() and np.isfinite(uncertainty).all() and np.all(uncertainty > 0)):
            raise ValueError("calibration values and sigmas must be finite; sigmas must be positive")

        def objective(viscosity):
            residual = (self._predict(experiments, float(viscosity)) - measured) / uncertainty
            return float(residual @ residual)

        result = minimize_scalar(objective, bounds=(0.006, 0.018), method="bounded",
                                 options={"xatol": 2e-10})
        self.viscosity = float(result.x)
        return self

    def _predict(self, experiments, viscosity):
        m, n, _, mean, *_ = operators()
        output = np.empty(len(experiments), dtype=float)
        for index, experiment in enumerate(experiments):
            bx, by = float(experiment["magnetic_x"]), float(experiment["magnetic_y"])
            if experiment["observable"] == "mean":
                observation = mean
            elif experiment["observable"] == "point":
                observation = (2.0 * np.sin(m * np.pi * float(experiment["x"]))
                               * np.sin(n * np.pi * float(experiment["y"])))
            else:
                raise ValueError("observable must be 'mean' or 'point'")
            response = 0.0
            for ids, rates, vectors, forcing in spectrum(viscosity, bx, by):
                transient = -np.expm1(-rates * float(experiment["time"])) / rates
                response += observation[ids] @ (vectors @ (transient * forcing))
            output[index] = float(experiment["pressure_gradient"]) / DENSITY * response
        return output

    def predict(self, experiments):
        if self.viscosity is None:
            raise RuntimeError("fit must be called before predict")
        return self._predict(list(experiments), self.viscosity)
