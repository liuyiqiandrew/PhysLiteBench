"""Transient Hartmann damping in a rectangular, insulating duct.

The velocity is expanded in no-slip sine modes.  At every magnetic field the
motional emf is first projected onto currents which are divergence free and
have zero normal component at the walls.  This is important for a tilted
field: the potential is shared by both parts of ``u cross B``.
"""

from functools import lru_cache

import numpy as np
from scipy.linalg import eigh
from scipy.optimize import minimize_scalar
from scipy.sparse import coo_matrix


WIDTH, HEIGHT = 0.012, 0.008
DENSITY, CONDUCTIVITY = 1000.0, 1.0e5
# The pressure starts in odd--odd modes.  A tilted field couples these to the
# even--even modes through its common electrostatic potential, so both are
# retained (1,120 modes total).
X_MODES, Y_MODES = 40, 56
POTENTIAL_MODES = 400


def _cos_sin_overlap(cos_index, sin_index):
    """Mean of cos(cos_index*pi*r) sin(sin_index*pi*r), 0 <= r <= 1."""
    c = np.asarray(cos_index, dtype=float)[..., None]
    s = np.asarray(sin_index, dtype=float)[None, ...]
    # Only the even-cosine/odd-sine combinations used below are nonzero.
    return 2.0 * s / (np.pi * (s * s - c * c))


@lru_cache(maxsize=1)
def operators():
    """Return modal data and the current-projection factors.

    The returned matrices include the weak Neumann-potential projection for
    both pieces of the motional emf, including their mixed term.
    """
    all_m = np.arange(1, X_MODES + 1, dtype=float)
    all_n = np.arange(1, Y_MODES + 1, dtype=float)
    m_grid, n_grid = np.meshgrid(all_m, all_n, indexing="ij")
    keep = (m_grid.astype(int) + n_grid.astype(int)) % 2 == 0
    m, n = m_grid[keep], n_grid[keep]
    count = m.size
    wave2 = (m * np.pi / WIDTH) ** 2 + (n * np.pi / HEIGHT) ** 2
    mean = np.where((m.astype(int) % 2 == 1) & (n.astype(int) % 2 == 1),
                    8.0 / (np.pi * np.pi * m * n), 0.0)

    # The potential has a cosine-cosine (Neumann) expansion.  Form sparse
    # matrices for <grad psi, (-u,0)> and <grad psi, (0,u)> respectively.
    # They have identical row numbering, which preserves their cross term.
    pvals = qvals = np.arange(POTENTIAL_MODES + 1, dtype=float)
    norm_p = np.where(pvals == 0.0, 1.0, np.sqrt(2.0))
    norm_q = np.where(qvals == 0.0, 1.0, np.sqrt(2.0))
    rows_x, cols_x, data_x = [], [], []
    rows_y, cols_y, data_y = [], [], []
    side = POTENTIAL_MODES + 1
    for column, (mm, nn) in enumerate(zip(m, n)):
        m_int, n_int = int(mm), int(nn)
        kxm, kyn = mm * np.pi / WIDTH, nn * np.pi / HEIGHT
        # x-gradient: its sine factor fixes p=m; q has opposite parity to n.
        q = qvals[(qvals.astype(int) + n_int) % 2 == 1]
        overlap = _cos_sin_overlap(q, np.array([nn])).ravel()
        factor = np.sqrt(2.0) * norm_q[q.astype(int)] * kxm * overlap
        factor /= np.sqrt(kxm * kxm + (q * np.pi / HEIGHT) ** 2)
        rows_x.extend((m_int * side + q.astype(int)).tolist())
        cols_x.extend([column] * len(q))
        data_x.extend(factor.tolist())
        # y-gradient: q=n; p has opposite parity to m.
        p = pvals[(pvals.astype(int) + m_int) % 2 == 1]
        overlap = _cos_sin_overlap(p, np.array([mm])).ravel()
        factor = -norm_p[p.astype(int)] * np.sqrt(2.0) * kyn * overlap
        factor /= np.sqrt((p * np.pi / WIDTH) ** 2 + kyn * kyn)
        rows_y.extend((p.astype(int) * side + n_int).tolist())
        cols_y.extend([column] * len(p))
        data_y.extend(factor.tolist())

    shape = (side * side, count)
    gx = coo_matrix((data_x, (rows_x, cols_x)), shape=shape).tocsr()
    gy = coo_matrix((data_y, (rows_y, cols_y)), shape=shape).tocsr()
    # These are dense modal matrices but are formed once.  The mixed matrix
    # is the oblique-field term lost by separate axis-response models.
    kxx = (gx.T @ gx).toarray()
    kyy = (gy.T @ gy).toarray()
    kxy = (gx.T @ gy + gy.T @ gx).toarray()
    return m, n, wave2, mean, kxx, kyy, kxy


@lru_cache(maxsize=128)
def spectrum(viscosity, bx, by):
    """Eigenpairs of the viscous plus charge-conserving Lorentz operator."""
    m, n, wave2, mean, kxx, kyy, kxy = operators()
    # j/sigma is the projection of u cross B.  The potential source is
    # By*gx + Bx*gy, so its squared norm includes Bx*By*kxy.
    magnetic = ((bx * bx + by * by) * np.eye(len(m))
                - by * by * kxx - bx * bx * kyy - bx * by * kxy)
    magnetic = (magnetic + magnetic.T) * 0.5
    matrix = np.diag(viscosity / DENSITY * wave2) + CONDUCTIVITY / DENSITY * magnetic
    rates, vectors = eigh(matrix, check_finite=False)
    return rates, vectors, vectors.T @ mean


class Model:
    def __init__(self):
        self.viscosity = None

    def fit(self, records):
        records = list(records)
        if not records:
            raise ValueError("at least one calibration record is required")
        values = np.asarray([r["value"] for r in records], dtype=float)
        errors = np.asarray([r["sigma"] for r in records], dtype=float)
        if not (np.isfinite(values).all() and np.isfinite(errors).all() and np.all(errors > 0)):
            raise ValueError("calibration values must be finite and sigma must be positive")
        records_input = [r["input"] for r in records]

        def objective(viscosity):
            residual = (self._predict(records_input, viscosity) - values) / errors
            return float(residual @ residual)

        result = minimize_scalar(
            objective, bounds=(0.006, 0.018), method="bounded",
            options={"xatol": 2e-9},
        )
        self.viscosity = float(np.clip(result.x, 0.006, 0.018))
        return self

    @staticmethod
    def _response(experiment, viscosity):
        m, n, _, mean, *_ = operators()
        bx = float(experiment["magnetic_x"])
        by = float(experiment["magnetic_y"])
        time = float(experiment["time"])
        rates, vectors, forcing = spectrum(float(viscosity), bx, by)
        coefficients = vectors @ ((-np.expm1(-rates * time) / rates) * forcing)
        if experiment["observable"] == "mean":
            observation = mean
        elif experiment["observable"] == "point":
            observation = 2.0 * np.sin(m * np.pi * float(experiment["x"])) * np.sin(
                n * np.pi * float(experiment["y"])
            )
        else:
            raise ValueError("observable must be 'mean' or 'point'")
        return float(experiment["pressure_gradient"]) / DENSITY * float(observation @ coefficients)

    def _predict(self, experiments, viscosity):
        return np.asarray([self._response(e, viscosity) for e in experiments], dtype=float)

    def predict(self, experiments):
        if self.viscosity is None:
            raise RuntimeError("fit must be called before predict")
        return self._predict(experiments, self.viscosity)
