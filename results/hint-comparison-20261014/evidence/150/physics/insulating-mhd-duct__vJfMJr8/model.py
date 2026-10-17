"""Transient duct-flow model with insulating electrical boundaries.

The velocity is expanded in no-slip sine modes. At every instant the
electric potential is eliminated from Ohm's law using its Neumann problem.
For an oblique field there is therefore one potential and one
divergence-free current, rather than two independently damped flows.
"""

from functools import lru_cache

import numpy as np
from scipy.linalg import eigh
from scipy.optimize import minimize_scalar


WIDTH, HEIGHT = 0.012, 0.008
DENSITY, CONDUCTIVITY = 1000.0, 1.0e5

# The constant pressure forcing only excites the even-under-centre-inversion
# subspace. An oblique field can mix its odd-odd and even-even members.
X_MODES, Y_MODES = 40, 56
POTENTIAL_MODES = 400


def _sine_cosine_overlap(sine, cosine):
    """Integral of normalized sine and cosine modes (rows are sine modes)."""
    sine = np.asarray(sine, dtype=int)[:, None]
    cosine = np.asarray(cosine, dtype=int)[None, :]
    total = sine + cosine
    difference = sine - cosine
    reciprocal_difference = np.divide(
        1.0, difference, out=np.zeros_like(total, dtype=float),
        where=difference != 0,
    )
    value = (1.0 / total + reciprocal_difference) / np.pi
    value *= total % 2 == 1
    # sqrt(2) for each nonconstant one-dimensional normalized mode.
    return value * np.sqrt(2.0) * np.where(cosine > 0, np.sqrt(2.0), 1.0)


@lru_cache(maxsize=1)
def operators(nx=X_MODES, ny=Y_MODES):
    """Build modal Laplacian, observables, and coupled magnetic damping."""
    mm, nn = np.meshgrid(np.arange(1, nx + 1), np.arange(1, ny + 1),
                         indexing="ij")
    keep = (mm + nn) % 2 == 0
    m, n = mm[keep], nn[keep]
    count = len(m)
    wave2 = (m * np.pi / WIDTH) ** 2 + (n * np.pi / HEIGHT) ** 2
    mean = np.where((m % 2 == 1) & (n % 2 == 1),
                    8.0 / (np.pi ** 2 * m * n), 0.0)

    # With a = u zhat x B = (-By*u, Bx*u), eliminating the potential gives
    # <P a_i, P a_j> = B^2 delta_ij - sum <a_i,grad psi><a_j,grad psi>/k^2.
    # psi are Neumann cosine modes and P is the zero-normal, solenoidal
    # projection. Build the two diagonal field pieces without a huge dense
    # potential-mode matrix.
    q_for_x = np.arange(POTENTIAL_MODES + 1)
    p_for_y = np.arange(POTENTIAL_MODES + 1)
    qxx = np.eye(count)
    qyy = np.eye(count)

    # By: potential modes have p=m and arbitrary q.
    for fixed_m in np.unique(m):
        ids = np.flatnonzero(m == fixed_m)
        coupling = ((fixed_m * np.pi / WIDTH)
                    * _sine_cosine_overlap(n[ids], q_for_x))
        denominator = ((fixed_m * np.pi / WIDTH) ** 2
                       + (q_for_x * np.pi / HEIGHT) ** 2)
        qxx[np.ix_(ids, ids)] -= (coupling / denominator) @ coupling.T

    # Bx: potential modes have q=n and arbitrary p.
    for fixed_n in np.unique(n):
        ids = np.flatnonzero(n == fixed_n)
        coupling = -((fixed_n * np.pi / HEIGHT)
                     * _sine_cosine_overlap(m[ids], p_for_y))
        denominator = ((p_for_y * np.pi / WIDTH) ** 2
                       + (fixed_n * np.pi / HEIGHT) ** 2)
        qyy[np.ix_(ids, ids)] -= (coupling / denominator) @ coupling.T

    # -Cx.T D Cy. Its transpose is the other half of the Bx*By term. Every
    # mixed entry selects one potential mode (p=m_i, q=n_j), so it has no
    # potential-mode truncation error.
    qcross = (
        (m[:, None] * np.pi / WIDTH)
        * (n[None, :] * np.pi / HEIGHT)
        * _sine_cosine_overlap(n, n)
        * _sine_cosine_overlap(m, m).T
        / ((m[:, None] * np.pi / WIDTH) ** 2
           + (n[None, :] * np.pi / HEIGHT) ** 2)
    )
    return m, n, wave2, mean, qxx, qyy, qcross


@lru_cache(maxsize=16)
def spectrum(viscosity, bx, by, nx=X_MODES, ny=Y_MODES):
    """Eigenpairs of the coupled viscous--magnetic decay operator."""
    _, _, wave2, mean, qxx, qyy, qcross = operators(nx, ny)
    magnetic = (by * by * qxx + bx * bx * qyy
                + bx * by * (qcross + qcross.T))
    matrix = np.diag(viscosity / DENSITY * wave2) + CONDUCTIVITY / DENSITY * magnetic
    matrix = 0.5 * (matrix + matrix.T)  # remove numerical asymmetry
    rates, vectors = eigh(matrix, check_finite=False)
    return rates, vectors, vectors.T @ mean


class Model:
    def __init__(self):
        self.viscosity = None

    def fit(self, records):
        records = list(records)
        if not records:
            raise ValueError("fit requires at least one calibration record")
        observed = np.asarray([record["value"] for record in records], dtype=float)
        uncertainty = np.asarray([record["sigma"] for record in records], dtype=float)
        if (not np.isfinite(observed).all() or not np.isfinite(uncertainty).all()
                or np.any(uncertainty <= 0)):
            raise ValueError("calibration values and positive sigmas must be finite")
        experiments = [record["input"] for record in records]

        def objective(viscosity):
            residual = (self._predict_at(float(viscosity), experiments) - observed) / uncertainty
            return float(residual @ residual)

        result = minimize_scalar(
            objective, bounds=(0.006, 0.018), method="bounded",
            options={"xatol": 1.0e-10},
        )
        self.viscosity = float(result.x)
        return self

    @staticmethod
    def _predict_at(viscosity, experiments):
        m, n, _, mean, *_ = operators()
        values = []
        for experiment in experiments:
            bx = float(experiment["magnetic_x"])
            by = float(experiment["magnetic_y"])
            pressure = float(experiment["pressure_gradient"])
            time = float(experiment["time"])
            rates, vectors, forcing = spectrum(viscosity, bx, by)
            modal = vectors @ ((-np.expm1(-rates * time) / rates) * forcing)
            if experiment["observable"] == "mean":
                observation = mean
            elif experiment["observable"] == "point":
                x, y = float(experiment["x"]), float(experiment["y"])
                observation = 2.0 * np.sin(m * np.pi * x) * np.sin(n * np.pi * y)
            else:
                raise ValueError("observable must be 'mean' or 'point'")
            values.append(pressure / DENSITY * float(observation @ modal))
        return np.asarray(values, dtype=float)

    def predict(self, experiments):
        if self.viscosity is None:
            raise ValueError("call fit before predict")
        return self._predict_at(float(self.viscosity), experiments)
