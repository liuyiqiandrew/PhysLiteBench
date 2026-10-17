"""Quasi-static MHD model for the rectangular insulating duct.

The velocity and the electric-current stream function are expanded in sine
series. This is convenient here: velocity has homogeneous Dirichlet
conditions and the current stream function is zero at every wall.
"""

from functools import lru_cache

import numpy as np
from scipy.linalg import eigh
from scipy.optimize import minimize_scalar


WIDTH, HEIGHT = 0.012, 0.008
DENSITY, CONDUCTIVITY = 1000.0, 1.0e5
X_MODES, Y_MODES = 40, 56
ELECTRIC_MODES = 400


def derivative_overlap(target, source):
    """<s_target, d s_source/dq> on the unit interval, for normalized sines."""
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
    """Return Galerkin operators in the pressure-driven parity sector.

    Uniform pressure forcing excites odd--odd velocity modes. An oblique
    field can additionally excite even--even modes, but never the other two
    parity sectors, hence the ``(m+n) even`` selection.
    """
    m, n = np.meshgrid(np.arange(1, nx + 1), np.arange(1, ny + 1), indexing="ij")
    keep = (m + n) % 2 == 0
    m, n = m[keep], n[keep]
    count = len(m)

    wave2 = (m * np.pi / WIDTH) ** 2 + (n * np.pi / HEIGHT) ** 2
    mean = np.where((m % 2 == 1) & (n % 2 == 1), 8.0 / (np.pi**2 * m * n), 0.0)
    qx = np.zeros((count, count))
    qy = np.zeros((count, count))
    electric = np.arange(1, ELECTRIC_MODES + 1)

    # M_ij = <d_i u, (-Delta)^-1 d_j u>. For a field B, magnetic
    # dissipation is (Bx*d_x + By*d_y)^T (-Delta)^-1 (...).
    for fixed_n in np.unique(n):
        ids = np.flatnonzero(n == fixed_n)
        d = derivative_overlap(electric, m[ids]) / WIDTH
        k2 = (electric * np.pi / WIDTH) ** 2 + (fixed_n * np.pi / HEIGHT) ** 2
        qx[np.ix_(ids, ids)] = d.T @ (d / k2[:, None])
    for fixed_m in np.unique(m):
        ids = np.flatnonzero(m == fixed_m)
        d = derivative_overlap(electric, n[ids]) / HEIGHT
        k2 = (fixed_m * np.pi / WIDTH) ** 2 + (electric * np.pi / HEIGHT) ** 2
        qy[np.ix_(ids, ids)] = d.T @ (d / k2[:, None])

    # This cross term couples odd--odd and even--even modes if both B
    # components are present. It is absent from axial-field calibration.
    dx = derivative_overlap(m, m).T / WIDTH  # velocity target, electric m
    dy = derivative_overlap(n, n) / HEIGHT   # electric n, velocity source
    electric_wave2 = (m[None, :] * np.pi / WIDTH) ** 2 + (n[:, None] * np.pi / HEIGHT) ** 2
    qxy = dx * dy / electric_wave2

    return m, n, wave2, mean, qx, qy, qxy


@lru_cache(maxsize=128)
def spectrum(viscosity, bx, by, nx=X_MODES, ny=Y_MODES):
    """Eigenmodes and force coefficients for one viscosity and field."""
    m, n, wave2, mean, qx, qy, qxy = operators(nx, ny)
    magnetic = bx * bx * qx + by * by * qy + bx * by * (qxy + qxy.T)
    operator = np.diag(viscosity / DENSITY * wave2) + CONDUCTIVITY / DENSITY * magnetic

    # Pressure does not force even--even modes for an axial field, so this
    # smaller block is exact and substantially speeds up fitting.
    if bx == 0.0 or by == 0.0:
        groups = [np.flatnonzero((m % 2 == 1) & (n % 2 == 1))]
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

    @staticmethod
    def _prediction(viscosity, experiments):
        m, n, _, mean, *_ = operators()
        output = np.empty(len(experiments), dtype=float)
        for index, experiment in enumerate(experiments):
            bx = float(experiment["magnetic_x"])
            by = float(experiment["magnetic_y"])
            time = float(experiment["time"])
            if experiment["observable"] == "mean":
                observation = mean
            else:
                observation = 2.0 * np.sin(m * np.pi * float(experiment["x"])) * np.sin(
                    n * np.pi * float(experiment["y"])
                )

            response = 0.0
            for ids, rates, vectors, forcing in spectrum(float(viscosity), bx, by):
                modal_step = -np.expm1(-rates * time) / rates
                response += observation[ids] @ (vectors @ (modal_step * forcing))
            output[index] = float(experiment["pressure_gradient"]) / DENSITY * response
        return output

    def fit(self, records):
        records = list(records)
        if not records:
            raise ValueError("at least one calibration record is required")
        experiments = [record["input"] for record in records]
        observed = np.asarray([record["value"] for record in records], dtype=float)
        uncertainty = np.asarray([record["sigma"] for record in records], dtype=float)
        if not (np.isfinite(observed).all() and np.isfinite(uncertainty).all() and (uncertainty > 0).all()):
            raise ValueError("calibration values must be finite and sigma must be positive")

        def chi_squared(viscosity):
            residual = (self._prediction(viscosity, experiments) - observed) / uncertainty
            return float(residual @ residual)

        optimum = minimize_scalar(
            chi_squared,
            bounds=(0.006, 0.018),
            method="bounded",
            options={"xatol": 1e-10},
        )
        self.viscosity = float(optimum.x)
        return self

    def predict(self, experiments):
        if self.viscosity is None:
            raise RuntimeError("fit must be called before predict")
        return self._prediction(self.viscosity, list(experiments))
