"""Transient MHD model for a conducting liquid in a rectangular duct."""

from functools import lru_cache

import numpy as np
from scipy.linalg import eigh
from scipy.optimize import minimize_scalar


WIDTH, HEIGHT = 0.012, 0.008
DENSITY, CONDUCTIVITY = 1000.0, 1.0e5
X_MODES, Y_MODES = 40, 56


def derivative_overlap(target, source):
    """Overlap of a normalized sine mode and a derivative of a sine mode."""
    p = np.asarray(target, dtype=float)[:, None]
    m = np.asarray(source, dtype=float)[None, :]
    denominator = p * p - m * m
    return np.divide(
        4.0 * p * m,
        denominator,
        out=np.zeros_like(denominator),
        where=((p + m) % 2 == 1),
    )


@lru_cache(maxsize=4)
def operators(nx=X_MODES, ny=Y_MODES):
    """Return the viscous and single-component magnetic operators."""
    m, n = np.meshgrid(
        np.arange(1, nx + 1), np.arange(1, ny + 1), indexing="ij"
    )
    keep = (m + n) % 2 == 0
    m, n = m[keep], n[keep]
    wave2 = (m * np.pi / WIDTH) ** 2 + (n * np.pi / HEIGHT) ** 2

    # This vector is both the uniform-force coefficient and the mean-velocity
    # observation coefficient in the adopted sine-series normalization.
    mean = np.where(m % 2 == 1, 8.0 / (np.pi * np.pi * m * n), 0.0)
    qx = np.zeros((len(m), len(m)))
    qy = np.zeros_like(qx)
    electric = np.arange(1, 401)

    # A Bx field damps variation across x (the motional emf is along y), and
    # vice versa.  Insulating walls make these operators block structured.
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
    return m, n, wave2, mean, qx, qy


@lru_cache(maxsize=4)
def cross_magnetic_operator(nx=X_MODES, ny=Y_MODES):
    """Symmetric magnetic coupling that multiplies ``Bx * By``.

    Both field components can generate the same electric-potential mode.  The
    resulting cross term is absent only for an axis-aligned field.
    """
    m, n, _, _, _, _ = operators(nx, ny)
    # C[p,m] is <normalised cos(p), normalised sin(m)> for positive modes.
    p = np.arange(1, nx + 1, dtype=float)[:, None]
    source_x = np.arange(1, nx + 1, dtype=float)[None, :]
    cx = np.divide(
        4.0 * source_x,
        np.pi * (source_x * source_x - p * p),
        out=np.zeros((nx, nx)),
        where=((p + source_x) % 2 == 1),
    )
    q = np.arange(1, ny + 1, dtype=float)[:, None]
    source_y = np.arange(1, ny + 1, dtype=float)[None, :]
    cy = np.divide(
        4.0 * source_y,
        np.pi * (source_y * source_y - q * q),
        out=np.zeros((ny, ny)),
        where=((q + source_y) % 2 == 1),
    )

    # For i=(m,n), j=(r,s), this is <D_y i, inv(-lap) D_x j>.
    # Its transpose supplies <D_x i, inv(-lap) D_y j>.
    mi, ni = m[:, None], n[:, None]
    rj, sj = m[None, :], n[None, :]
    denominator = (rj * np.pi / WIDTH) ** 2 + (ni * np.pi / HEIGHT) ** 2
    first = (
        (ni * np.pi / HEIGHT)
        * (rj * np.pi / WIDTH)
        / denominator
        * cx[rj.astype(int) - 1, mi.astype(int) - 1]
        * cy[ni.astype(int) - 1, sj.astype(int) - 1]
    )
    return first + first.T


@lru_cache(maxsize=12)
def spectrum(viscosity, bx, by, nx=X_MODES, ny=Y_MODES):
    """Modal damping rates and force projections for one magnetic setting."""
    m, n, wave2, mean, qx, qy = operators(nx, ny)
    base = np.diag(viscosity / DENSITY * wave2)
    magnetic = bx * bx * qx + by * by * qy

    if bx != 0.0 and by != 0.0:
        magnetic = magnetic + bx * by * cross_magnetic_operator(nx, ny)
        rates, vectors = eigh(
            base + CONDUCTIVITY / DENSITY * magnetic, check_finite=False
        )
        ids = np.arange(len(m))
        return ((ids, rates, vectors, vectors.T @ mean),)

    # For an axis-aligned field the unforced even/even blocks can be skipped.
    if bx == 0.0:
        groups = [
            np.flatnonzero((m == fixed) & (n % 2 == 1))
            for fixed in np.unique(m[m % 2 == 1])
        ]
    else:
        groups = [
            np.flatnonzero((n == fixed) & (m % 2 == 1))
            for fixed in np.unique(n[n % 2 == 1])
        ]
    operator = base + CONDUCTIVITY / DENSITY * magnetic
    result = []
    for ids in groups:
        rates, vectors = eigh(operator[np.ix_(ids, ids)], check_finite=False)
        result.append((ids, rates, vectors, vectors.T @ mean[ids]))
    return tuple(result)


class Model:
    def __init__(self):
        self.viscosity = None

    @staticmethod
    def _predict(experiments, viscosity):
        m, n, _, mean, _, _ = operators()
        output = np.empty(len(experiments), dtype=float)
        for index, experiment in enumerate(experiments):
            bx = float(experiment["magnetic_x"])
            by = float(experiment["magnetic_y"])
            time = float(experiment["time"])
            if experiment["observable"] == "mean":
                observation = mean
            else:
                observation = (
                    2.0
                    * np.sin(m * np.pi * float(experiment["x"]))
                    * np.sin(n * np.pi * float(experiment["y"]))
                )
            response = 0.0
            for ids, rates, vectors, forcing in spectrum(viscosity, bx, by):
                amplitudes = vectors @ (
                    -np.expm1(-rates * time) / rates * forcing
                )
                response += observation[ids] @ amplitudes
            output[index] = float(experiment["pressure_gradient"]) / DENSITY * response
        return output

    def fit(self, records):
        """Estimate viscosity by Gaussian weighted least squares."""
        records = list(records)
        if not records:
            raise ValueError("at least one calibration record is required")
        experiments = [record["input"] for record in records]
        measured = np.asarray([record["value"] for record in records], dtype=float)
        uncertainty = np.asarray([record["sigma"] for record in records], dtype=float)
        if (
            not np.isfinite(measured).all()
            or not np.isfinite(uncertainty).all()
            or np.any(uncertainty <= 0.0)
        ):
            raise ValueError("calibration values and sigmas must be finite, with positive sigma")

        def loss(viscosity):
            residual = (self._predict(experiments, float(viscosity)) - measured) / uncertainty
            return float(residual @ residual)

        optimum = minimize_scalar(
            loss,
            bounds=(0.006, 0.018),
            method="bounded",
            options={"xatol": 1.0e-9},
        )
        if not optimum.success or not np.isfinite(optimum.fun):
            candidates = np.array([0.006, 0.012, 0.018])
            self.viscosity = float(candidates[np.argmin([loss(v) for v in candidates])])
        else:
            self.viscosity = float(np.clip(optimum.x, 0.006, 0.018))
        return self

    def predict(self, experiments):
        if self.viscosity is None:
            raise RuntimeError("fit must be called before predict")
        return self._predict(list(experiments), float(self.viscosity))
