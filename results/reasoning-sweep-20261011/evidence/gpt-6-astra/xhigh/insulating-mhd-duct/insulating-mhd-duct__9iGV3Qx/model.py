"""Transient, quasistatic MHD flow in a duct with insulating walls.

Write the transverse current as j = (psi_y, -psi_x). Charge conservation
and insulation then give psi = 0 on the boundary and

    -laplacian(psi) = sigma * (Bx * u_x + By * u_y).

Eliminating psi from rho*u_t = g + eta*laplacian(u) + B.grad(psi)
gives a positive, self-adjoint decay operator. We discretize it in a
no-slip sine basis and integrate the constant pressure forcing exactly in
time. In particular, the magnetic operator includes the Bx*By terms.
"""

from functools import lru_cache

import numpy as np
from scipy.linalg import eigh
from scipy.optimize import minimize_scalar
from scipy.special import erfc


WIDTH, HEIGHT = 0.012, 0.008
DENSITY, CONDUCTIVITY = 1000.0, 1e5
X_MODES, Y_MODES = 56, 40

# Integrate diffusion over s = time*r**2. This change of variables removes
# the square-root boundary-layer singularity in the mean at startup.
_quadrature_nodes, _quadrature_weights = np.polynomial.legendre.leggauss(64)
_quadrature_nodes = (_quadrature_nodes + 1) / 2
_quadrature_weights = _quadrature_weights * _quadrature_nodes


def _diffusive_survival(theta, position):
    """Unit initial temperature in a unit interval with zero end temperatures."""
    result = np.empty_like(theta)
    short = theta < 0.01
    if position is None:
        result[short] = 1 - 4 * np.sqrt(theta[short] / np.pi)
    else:
        scale = 2 * np.sqrt(theta[short])
        result[short] = (
            1 - erfc(position / scale) - erfc((1 - position) / scale)
            + erfc((1 + position) / scale) + erfc((2 - position) / scale)
        )
    odd = np.arange(1, 50, 2)
    coefficient = (
        8 / (np.pi * odd) ** 2
        if position is None else 4 * np.sin(np.pi * odd * position) / (np.pi * odd)
    )
    result[~short] = np.exp(-np.pi**2 * theta[~short, None] * odd**2) @ coefficient
    return result


@lru_cache(maxsize=1024)
def _unmagnetized_response(viscosity, time, x=None, y=None):
    """Untruncated rectangular-duct response to unit acceleration."""
    if time == 0:
        return 0.0
    diffusion_time = viscosity / DENSITY * time * _quadrature_nodes**2
    sx = _diffusive_survival(diffusion_time / WIDTH**2, x)
    sy = _diffusive_survival(diffusion_time / HEIGHT**2, y)
    return float(time * (_quadrature_weights @ (sx * sy)))


def derivative_overlap(target, source):
    """Sine/derivative overlap on a unit interval (area-normalized basis)."""
    p = np.asarray(target, dtype=float)[:, None]
    m = np.asarray(source, dtype=float)[None, :]
    return np.divide(
        4 * p * m,
        p * p - m * m,
        out=np.zeros((len(p), m.shape[1])),
        where=(p + m) % 2 == 1,
    )


def _directional_block(modes, length, transverse_wave):
    """Exact matrix of derivative* (-laplacian_D)^-1 derivative.

    Solving the one-dimensional Dirichlet Helmholtz equation analytically
    includes all current modes, including those above the velocity cutoff.
    The supplied velocity modes all have the same parity.
    """
    wave = modes * np.pi / length
    wave2 = wave * wave + transverse_wave * transverse_wave
    boundary_factor = np.tanh(transverse_wave * length / 2)
    if modes[0] % 2:
        boundary_factor = 1 / boundary_factor
    return np.diag(wave * wave / wave2) - (
        4 * transverse_wave / length * boundary_factor
    ) * np.outer(wave / wave2, wave / wave2)


@lru_cache(maxsize=3)
def operators(nx=X_MODES, ny=Y_MODES):
    """Galerkin operators, with inner product equal to cross-sectional mean."""
    m, n = np.meshgrid(np.arange(1, nx + 1), np.arange(1, ny + 1), indexing="ij")
    # Central inversion preserves the solution. Oblique fields couple the
    # odd/odd and even/even modes, so neither family may be discarded.
    keep = (m + n) % 2 == 0
    m, n = m[keep], n[keep]
    wave2 = (m * np.pi / WIDTH) ** 2 + (n * np.pi / HEIGHT) ** 2
    mean = np.where(m % 2 == 1, 8 / (np.pi**2 * m * n), 0.0)
    qx = np.zeros((len(m), len(m)))
    qy = np.zeros_like(qx)
    for fixed in np.unique(n):
        ids = np.flatnonzero(n == fixed)
        qx[np.ix_(ids, ids)] = _directional_block(
            m[ids], WIDTH, fixed * np.pi / HEIGHT
        )
    for fixed in np.unique(m):
        ids = np.flatnonzero(m == fixed)
        qy[np.ix_(ids, ids)] = _directional_block(
            n[ids], HEIGHT, fixed * np.pi / WIDTH
        )

    # <Dx basis_a, (-laplacian_D)^-1 Dy basis_b> has just one
    # overlapping current mode: (m_b, n_a). Add its transpose as well.
    cross = (
        derivative_overlap(m, m).T
        * derivative_overlap(n, n)
        / (WIDTH * HEIGHT)
        / ((m[None, :] * np.pi / WIDTH) ** 2 + (n[:, None] * np.pi / HEIGHT) ** 2)
    )
    cross += cross.T.copy()
    return m, n, wave2, mean, qx, qy, cross


@lru_cache(maxsize=12)
def spectrum(viscosity, bx, by, nx=X_MODES, ny=Y_MODES):
    """Decay modes and pressure-forcing projections for one preparation."""
    m, n, wave2, mean, qx, qy, cross = operators(nx, ny)
    if bx == 0 and by == 0:
        ids = np.flatnonzero(mean)
        return [(ids, viscosity / DENSITY * wave2[ids], None, mean[ids])]

    magnetic = bx * bx * qx + by * by * qy + bx * by * cross
    operator = CONDUCTIVITY / DENSITY * magnetic
    operator.flat[:: len(m) + 1] += viscosity / DENSITY * wave2
    if bx == 0:
        groups = [
            np.flatnonzero((m == fixed) & (n % 2 == 1))
            for fixed in np.unique(m[m % 2 == 1])
        ]
    elif by == 0:
        groups = [
            np.flatnonzero((n == fixed) & (m % 2 == 1))
            for fixed in np.unique(n[n % 2 == 1])
        ]
    else:
        groups = [np.arange(len(m))]
    result = []
    for ids in groups:
        rates, vectors = eigh(operator[np.ix_(ids, ids)], check_finite=False)
        result.append((ids, rates, vectors, vectors.T @ mean[ids]))
    return result


class Model:
    def __init__(self):
        self.viscosity = None

    def fit(self, records):
        """Estimate viscosity by Gaussian maximum likelihood within its bounds."""
        records = list(records)
        if not records:
            raise ValueError("At least one calibration record is required.")
        experiments = [record["input"] for record in records]
        measured = np.asarray([record["value"] for record in records], dtype=float)
        sigma = np.asarray([record["sigma"] for record in records], dtype=float)
        if not np.all(np.isfinite(measured)) or not np.all(np.isfinite(sigma) & (sigma > 0)):
            raise ValueError("Calibration values must be finite and uncertainties positive.")

        def objective(viscosity):
            residual = (self._predict(experiments, viscosity) - measured) / sigma
            return float(residual @ residual)

        result = minimize_scalar(
            objective,
            bounds=(0.006, 0.018),
            method="bounded",
            options={"xatol": 1e-10},
        )
        candidates = [(result.fun, result.x)]
        candidates.extend((objective(eta), eta) for eta in (0.006, 0.018))
        self.viscosity = float(min(candidates)[1])
        return self

    def predict(self, experiments):
        """Return mean or point velocities in m/s, in input order."""
        if self.viscosity is None:
            raise ValueError("Fit the model before predicting.")
        return self._predict(experiments, self.viscosity)

    @staticmethod
    def _predict(experiments, viscosity):
        experiments = list(experiments)
        output = np.zeros(len(experiments), dtype=float)
        m, n, wave2, mean, *_ = operators()
        odd = np.flatnonzero(mean)
        unmagnetized_rates = viscosity / DENSITY * wave2[odd]
        unmagnetized_responses = {}
        groups = {}
        for i, experiment in enumerate(experiments):
            if experiment["time"] == 0 or experiment["pressure_gradient"] == 0:
                continue
            bx, by = experiment["magnetic_x"], experiment["magnetic_y"]
            # Reversing the entire field leaves the Lorentz force unchanged.
            if bx < 0 or (bx == 0 and by < 0):
                bx, by = -bx, -by
            groups.setdefault((bx, by), []).append((i, experiment))

        for (bx, by), entries in groups.items():
            modes = spectrum(viscosity, bx, by)
            projections = {}
            responses = {}
            for i, experiment in entries:
                observable = experiment["observable"]
                if observable == "mean":
                    key = ("mean",)
                    observation = mean
                elif observable == "point":
                    key = ("point", experiment["x"], experiment["y"])
                    observation = 2 * np.sin(m * np.pi * key[1]) * np.sin(n * np.pi * key[2])
                else:
                    raise ValueError("Observable must be 'mean' or 'point'.")
                if key not in projections:
                    projections[key] = [
                        (observation[ids] if vectors is None else observation[ids] @ vectors) * forcing
                        for ids, _, vectors, forcing in modes
                    ]
                time = experiment["time"]
                if time not in responses:
                    responses[time] = [
                        -np.expm1(-rates * time) / rates for _, rates, _, _ in modes
                    ]
                value = sum(
                    weight @ response
                    for weight, response in zip(projections[key], responses[time])
                )
                # Restore the analytically known diffusion tail. Magnetic
                # damping is bounded, whereas viscous damping grows as k**2,
                # so unresolved high modes are dominated by plain diffusion.
                # This also recovers the exact initial interior acceleration.
                if time not in unmagnetized_responses:
                    unmagnetized_responses[time] = (
                        -np.expm1(-unmagnetized_rates * time) / unmagnetized_rates
                    )
                unmagnetized_finite = (
                    observation[odd] * mean[odd]
                ) @ unmagnetized_responses[time]
                position = () if observable == "mean" else key[1:]
                value += _unmagnetized_response(viscosity, time, *position) - unmagnetized_finite
                output[i] = experiment["pressure_gradient"] / DENSITY * value
        return output
