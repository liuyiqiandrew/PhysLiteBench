"""Bulk, electroneutral model for the electrolyte cell.

The state used by TransportModel is the pair of total A and B constituent
concentrations. Free concentrations and the neutral cluster are eliminated
locally using the fast association equilibrium.
"""

import numpy as np
from scipy.fft import dct, idct
from scipy.integrate import solve_ivp
from scipy.linalg import expm
from scipy.optimize import minimize_scalar
from scipy.sparse import diags


# A+ + B+ + 2 C- <-> P
ASSOCIATION = 0.5
NEUTRAL_MOBILITY = 0.15
CELL_LENGTH = 1.0e-3


def equilibrium(totals):
    """Return free_A, free_B, free_C, and P at local equilibrium.

    Totals has a final dimension of length two. The scalar equation for P is
    monotone on its physical interval, so safeguarded Newton iteration is
    quick and well behaved near pure preparations.
    """
    totals = np.asarray(totals, dtype=float)
    if totals.shape[-1:] != (2,):
        raise ValueError("totals must have a final dimension of length 2")

    # Clipping also keeps implicit-solver trial states from producing an
    # invalid mass-action evaluation through a tiny negative concentration.
    a = np.maximum(totals[..., 0], 0.0)
    b = np.maximum(totals[..., 1], 0.0)
    upper = np.minimum(a, b)
    p = np.zeros_like(a)

    for _ in range(40):
        af = a - p
        bf = b - p
        cf = af + bf
        residual = p - ASSOCIATION * af * bf * cf**2
        derivative = 1.0 + ASSOCIATION * (cf**3 + 4.0 * af * bf * cf)
        step = residual / derivative
        next_p = np.clip(p - step, 0.0, upper)
        if np.max(np.abs(next_p - p), initial=0.0) < 2.0e-14:
            p = next_p
            break
        p = next_p

    af = a - p
    bf = b - p
    return np.stack((af, bf, af + bf, p), axis=-1)


def _validate_mesh(x):
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or x.size < 2:
        raise ValueError("x must be a one-dimensional mesh with at least two points")
    dx = np.diff(x)
    if not np.all(np.isfinite(x)) or not np.all(dx > 0.0):
        raise ValueError("x must be finite and strictly increasing")
    if not np.allclose(dx, dx[0], rtol=2e-11, atol=1e-14):
        raise ValueError("x must be equally spaced")
    return x, dx[0]


def _validate_times(t):
    t = np.asarray(t, dtype=float)
    if t.ndim != 1 or t.size == 0:
        raise ValueError("t must be a nonempty one-dimensional array")
    if not np.all(np.isfinite(t)) or np.any(t < 0.0) or np.any(np.diff(t) < 0.0):
        raise ValueError("t must contain increasing, nonnegative finite times")
    return t


class TransportModel:
    def __init__(self):
        self.diffusivity = None

    @staticmethod
    def _binary(t, x, initial, diffusivity):
        """Exact semidiscrete solution for a pure A or pure B preparation.

        Zero current makes the effective diffusivities 4D/3 (A/C) and 8D/3
        (B/C). The DCT-I modes are the no-flux modes of the endpoint
        half-control-volume discretization.
        """
        t = np.asarray(t, dtype=float)
        x, dx = _validate_mesh(x)
        initial = np.asarray(initial, dtype=float)
        if initial.shape != (x.size, 2):
            raise ValueError("initial must have shape (len(x), 2)")

        intervals = x.size - 1
        wave_number = np.arange(x.size)
        laplacian_eigenvalues = (
            4.0 / dx**2 * np.sin(wave_number * np.pi / (2.0 * intervals)) ** 2
        )
        modes = dct(initial, type=1, axis=0)
        rates = laplacian_eigenvalues[:, None] * diffusivity * np.array(
            [4.0 / 3.0, 8.0 / 3.0]
        )[None, :]
        return idct(
            np.exp(-t[:, None, None] * rates[None, :, :]) * modes[None, :, :],
            type=1,
            axis=1,
        )

    def fit(self, data):
        """Fit the shared ionic diffusion scale by weighted least squares."""
        try:
            t = _validate_times(data["t"])
            x, _ = _validate_mesh(data["x"])
            initial = np.asarray(data["initial"], dtype=float)
            observed = np.asarray(data["concentration"], dtype=float)
            sigma = np.asarray(data["sigma"], dtype=float)
        except (KeyError, TypeError) as exc:
            raise ValueError("data must contain t, x, initial, concentration, and sigma") from exc

        if initial.ndim != 3 or initial.shape[1:] != (x.size, 2):
            raise ValueError("initial must have shape (records, len(x), 2)")
        expected = (initial.shape[0], t.size, x.size, 2)
        if observed.shape != expected or sigma.shape != expected:
            raise ValueError("concentration and sigma must match the documented calibration shape")
        if (
            not np.all(np.isfinite(initial))
            or not np.all(np.isfinite(observed))
            or not np.all(np.isfinite(sigma))
            or np.any(sigma <= 0.0)
        ):
            raise ValueError("calibration arrays must be finite and sigma must be positive")

        # Calibration preparations are binary electrolytes. Their analytic
        # propagation avoids numerical fitting bias in the single unknown D.
        def objective(log_diffusivity):
            diffusivity = float(np.exp(log_diffusivity))
            predicted = np.empty_like(observed)
            for record, profile in enumerate(initial):
                predicted[record] = self._binary(t, x, profile, diffusivity)
            residual = (predicted - observed) / sigma
            return float(np.sum(residual * residual))

        result = minimize_scalar(
            objective,
            bounds=(np.log(1.0e-14), np.log(1.0e-5)),
            method="bounded",
            options={"xatol": 1.0e-11},
        )
        if not result.success or not np.isfinite(result.fun):
            raise RuntimeError("could not fit the diffusion coefficient")
        self.diffusivity = float(np.exp(result.x))
        return self

    def predict(self, t, x, initial):
        """Predict total A and B concentrations at the requested times."""
        if self.diffusivity is None or not np.isfinite(self.diffusivity) or self.diffusivity <= 0.0:
            raise RuntimeError("fit must be called before predict")
        t = _validate_times(t)
        x, dx = _validate_mesh(x)
        initial = np.asarray(initial, dtype=float)
        if initial.shape != (x.size, 2):
            raise ValueError("initial must have shape (len(x), 2)")
        if not np.all(np.isfinite(initial)) or np.any(initial < 0.0):
            raise ValueError("initial concentrations must be finite and nonnegative")

        # No B everywhere (or no A everywhere) implies no cluster anywhere.
        if np.all(initial[:, 0] == 0.0) or np.all(initial[:, 1] == 0.0):
            return self._binary(t, x, initial, self.diffusivity)
        if t[-1] == 0.0:
            return np.broadcast_to(initial, (t.size, x.size, 2)).copy()

        n_points = x.size
        widths = np.full(n_points, dx)
        widths[[0, -1]] *= 0.5
        d_a, d_b, d_c = self.diffusivity * np.array([1.0, 4.0, 2.0])
        d_p = self.diffusivity * NEUTRAL_MOBILITY

        def rhs(_time, flat_totals):
            totals = flat_totals.reshape(n_points, 2)
            species = equilibrium(totals)
            gradient = np.diff(species, axis=0) / dx
            face = 0.5 * (species[:-1] + species[1:])

            # This field gradient enforces J_A + J_B - J_C = 0 face by face.
            numerator = d_a * gradient[:, 0] + d_b * gradient[:, 1] - d_c * gradient[:, 2]
            denominator = d_a * face[:, 0] + d_b * face[:, 1] + d_c * face[:, 2]
            electric_gradient = np.divide(
                -numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0.0
            )
            free_flux = -np.column_stack(
                (
                    d_a * (gradient[:, 0] + face[:, 0] * electric_gradient),
                    d_b * (gradient[:, 1] + face[:, 1] * electric_gradient),
                )
            )
            cluster_flux = -d_p * gradient[:, 3]

            # Fluxes of each conserved cation constituent; outer faces are
            # zero, implementing the blocking boundary condition.
            total_flux = np.zeros((n_points + 1, 2))
            total_flux[1:-1] = free_flux + cluster_flux[:, None]
            return (-np.diff(total_flux, axis=0) / widths[:, None]).ravel()

        # The nonlinear endpoint half-cell system is mildly stiff. The local
        # block-tridiagonal sparsity avoids dense finite-difference Jacobians.
        offsets = np.arange(-3, 4)
        sparsity = diags(
            [np.ones(2 * n_points - abs(offset)) for offset in offsets], offsets, format="csr"
        )
        solution = solve_ivp(
            rhs,
            (0.0, float(t[-1])),
            initial.ravel(),
            method="BDF",
            t_eval=t,
            rtol=2.0e-8,
            atol=2.0e-10,
            jac_sparsity=sparsity,
        )
        if not solution.success:
            raise RuntimeError(solution.message)
        answer = solution.y.T.reshape(t.size, n_points, 2)
        # Solver tolerance can leave a harmless negative roundoff at depletion.
        return np.maximum(answer, 0.0)

    def correlate(self, t, totals, harmonic=1):
        """Return the normalized two-time covariance of one cosine mode."""
        if self.diffusivity is None or not np.isfinite(self.diffusivity) or self.diffusivity <= 0.0:
            raise RuntimeError("fit must be called before correlate")
        t = _validate_times(t)
        totals = np.asarray(totals, dtype=float)
        if totals.shape != (2,) or not np.all(np.isfinite(totals)) or np.any(totals <= 0.0):
            raise ValueError("totals must be a finite, positive pair (A_total, B_total)")
        if isinstance(harmonic, (bool, np.bool_)) or int(harmonic) != harmonic or not 1 <= int(harmonic) <= 8:
            raise ValueError("harmonic must be an integer from 1 through 8")
        return equilibrium_correlations(t, totals, int(harmonic), self.diffusivity)


def fluctuation_matrices(totals, diffusivity):
    """Thermodynamic susceptibility and Onsager mobility of total A/B."""
    concentration = equilibrium(np.asarray(totals, dtype=float))
    if concentration.shape != (4,):
        raise ValueError("totals must be a pair")

    charge = np.array([1.0, 1.0, -1.0, 0.0])
    constituents = np.array([[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0]])
    population = np.diag(concentration)
    charge_variance = float(charge @ population @ charge)
    conditional_population = population - np.outer(population @ charge, population @ charge) / charge_variance
    susceptibility = constituents @ conditional_population @ constituents.T

    species_mobility = np.diag(
        diffusivity * np.array([1.0, 4.0, 2.0, NEUTRAL_MOBILITY]) * concentration
    )
    mobile_charge = species_mobility @ charge
    constrained_mobility = species_mobility - np.outer(mobile_charge, mobile_charge) / float(
        charge @ mobile_charge
    )
    mobility = constituents @ constrained_mobility @ constituents.T
    return susceptibility, mobility


def equilibrium_correlations(t, totals, harmonic, diffusivity):
    """Linearized, stationary mode covariance at nonnegative lags."""
    susceptibility, mobility = fluctuation_matrices(totals, diffusivity)
    wave_number_squared = (harmonic * np.pi / CELL_LENGTH) ** 2
    relaxation = wave_number_squared * np.linalg.solve(susceptibility.T, mobility.T).T

    # The detector convention supplies independent ideal total-constituent
    # fluctuations at equal time; the nonzero-mode factor of two is included.
    equal_time = np.diag(np.asarray(totals, dtype=float))
    return np.asarray([expm(-float(lag) * relaxation) @ equal_time for lag in t])
