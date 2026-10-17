"""Transport model for the electrolyte cell described in :mod:`README`.

The state used by ``predict`` is the pair of *total* constituent
concentrations.  Free concentrations are recovered locally from mass action;
this avoids introducing an artificial reaction time scale into the transport
calculation.
"""

from __future__ import annotations

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


def equilibrium(total: np.ndarray) -> np.ndarray:
    """Return ``(a, b, c, p)`` at local equilibrium for total ``(A, B)``.

    The scalar equation for the bound concentration is monotone on
    ``[0, min(A, B)]``.  Bisection is a little less clever than a raw Newton
    iteration, but remains well behaved at a pure-component interface where
    the free concentration of one ion vanishes.
    """
    total = np.asarray(total, dtype=float)
    if total.shape[-1:] != (2,):
        raise ValueError("total concentrations must have a final dimension of length 2")

    # Valid preparations are non-negative.  The clipping also keeps an
    # implicit ODE solver's occasional trial state from producing NaNs.
    a_total = np.maximum(total[..., 0], 0.0)
    b_total = np.maximum(total[..., 1], 0.0)
    lower = np.zeros(np.broadcast(a_total, b_total).shape, dtype=float)
    upper = np.minimum(a_total, b_total).copy()

    # f(p) = p - K (A-p)(B-p)(A+B-2p)^2 is strictly increasing in its
    # physical interval.  54 iterations is below double precision interval
    # resolution for the concentration range used by the apparatus.
    for _ in range(54):
        bound = 0.5 * (lower + upper)
        free_a = a_total - bound
        free_b = b_total - bound
        anion = free_a + free_b
        value = bound - ASSOCIATION * free_a * free_b * anion**2
        upper = np.where(value > 0.0, bound, upper)
        lower = np.where(value > 0.0, lower, bound)

    bound = 0.5 * (lower + upper)
    free_a = a_total - bound
    free_b = b_total - bound
    return np.stack((free_a, free_b, free_a + free_b, bound), axis=-1)


def _check_grid(x: np.ndarray) -> float:
    """Validate the nodal mesh and return its spacing."""
    if x.ndim != 1 or x.size < 2 or not np.isfinite(x).all():
        raise ValueError("x must be a finite one-dimensional mesh with at least two points")
    spacing = np.diff(x)
    dx = float(spacing[0])
    if dx <= 0.0 or not np.allclose(spacing, dx, rtol=2e-12, atol=1e-15):
        raise ValueError("x must be equally spaced in increasing order")
    return dx


def _check_times(t: np.ndarray) -> np.ndarray:
    t = np.asarray(t, dtype=float)
    if t.ndim != 1 or t.size == 0 or not np.isfinite(t).all():
        raise ValueError("t must be a non-empty finite one-dimensional array")
    if t[0] < 0.0 or np.any(np.diff(t) <= 0.0):
        # A single zero is, of course, permitted.
        if not (t.size == 1 and t[0] == 0.0):
            raise ValueError("t must contain increasing non-negative times")
    return t


class TransportModel:
    """Fit and evaluate the fast-equilibrium electrolyte transport model."""

    def __init__(self):
        self.diffusivity: float | None = None

    def _require_fit(self) -> float:
        if self.diffusivity is None or not np.isfinite(self.diffusivity) or self.diffusivity <= 0.0:
            raise RuntimeError("call fit before requesting a prediction")
        return float(self.diffusivity)

    def fit(self, data):
        """Fit the common ion diffusion scale by weighted least squares.

        Calibration preparations are binary by construction.  Their exact
        zero-current ambipolar diffusivities are 4D/3 (A/C) and 8D/3 (B/C),
        so a DCT-I diagonalizes the same finite-volume no-flux operator used
        by the general solver.  This makes fitting fast and avoids numerical
        ODE error from influencing the one fitted parameter.
        """
        try:
            t = _check_times(np.asarray(data["t"], dtype=float))
            x = np.asarray(data["x"], dtype=float)
            _check_grid(x)
            initial = np.asarray(data["initial"], dtype=float)
            observed = np.asarray(data["concentration"], dtype=float)
            sigma = np.asarray(data["sigma"], dtype=float)
        except (KeyError, TypeError) as exc:
            raise ValueError("data must contain x, t, initial, concentration, and sigma") from exc

        expected_initial_shape = (initial.shape[0], x.size, 2) if initial.ndim == 3 else None
        expected_observed_shape = (initial.shape[0], t.size, x.size, 2)
        if (
            expected_initial_shape is None
            or initial.shape != expected_initial_shape
            or observed.shape != expected_observed_shape
            or sigma.shape != expected_observed_shape
        ):
            raise ValueError("calibration array shapes do not match x and t")
        if not (np.isfinite(initial).all() and np.isfinite(observed).all() and np.isfinite(sigma).all()):
            raise ValueError("calibration arrays must be finite")
        if np.any(initial < 0.0) or np.any(sigma <= 0.0):
            raise ValueError("initial concentrations must be non-negative and sigma positive")

        binary = np.all(np.isclose(initial[..., 0], 0.0), axis=1) | np.all(
            np.isclose(initial[..., 1], 0.0), axis=1
        )

        def objective(log_diffusivity: float) -> float:
            self.diffusivity = float(np.exp(log_diffusivity))
            if np.all(binary):
                predicted = np.stack([self._binary(t, x, state) for state in initial])
            else:
                # Not used by the supplied calibration protocol, but retains
                # a sensible fit definition if an extended data set is used.
                predicted = np.stack([self.predict(t, x, state) for state in initial])
            residual = (predicted - observed) / sigma
            return float(np.sum(residual * residual))

        # A deliberately broad physical interval.  Optimising log(D) keeps
        # the parameter positive and scales the one-dimensional search well.
        result = minimize_scalar(
            objective,
            bounds=(np.log(1.0e-14), np.log(1.0e-5)),
            method="bounded",
            options={"xatol": 2.0e-12},
        )
        if not result.success or not np.isfinite(result.fun):
            self.diffusivity = None
            raise RuntimeError("unable to fit a finite diffusivity")
        self.diffusivity = float(np.exp(result.x))
        return self

    def _binary(self, t: np.ndarray, x: np.ndarray, initial: np.ndarray) -> np.ndarray:
        """Exact semidiscrete solution for a preparation containing A or B."""
        diffusivity = self._require_fit()
        t = np.asarray(t, dtype=float)
        x = np.asarray(x, dtype=float)
        initial = np.asarray(initial, dtype=float)
        intervals = x.size - 1
        dx = x[1] - x[0]

        # Eigenvalues of the node-centred finite-volume Neumann Laplacian.
        wave_number = np.arange(intervals + 1)
        laplacian_rate = 4.0 * np.sin(wave_number * np.pi / (2.0 * intervals)) ** 2 / dx**2
        modes = dct(initial, type=1, axis=0)
        ambipolar = diffusivity * np.array((4.0 / 3.0, 8.0 / 3.0))
        decay = np.exp(-t[:, None, None] * laplacian_rate[None, :, None] * ambipolar[None, None, :])
        return idct(decay * modes[None, :, :], type=1, axis=1)

    def predict(self, t, x, initial):
        """Predict total A and B constituent profiles at the requested times."""
        diffusivity = self._require_fit()
        t = _check_times(np.asarray(t, dtype=float))
        x = np.asarray(x, dtype=float)
        dx = _check_grid(x)
        initial = np.asarray(initial, dtype=float)
        if initial.shape != (x.size, 2) or not np.isfinite(initial).all() or np.any(initial < 0.0):
            raise ValueError("initial must be a finite non-negative (len(x), 2) array")
        if t[-1] == 0.0:
            return initial[None, :, :].copy()

        # A single component has no clusters and admits the exact linear
        # solution.  This path is also important for the calibration fit.
        if np.all(initial[:, 0] == 0.0) or np.all(initial[:, 1] == 0.0):
            return self._binary(t, x, initial)

        nodes = x.size
        control_volume_width = np.full(nodes, dx)
        control_volume_width[[0, -1]] = 0.5 * dx
        ion_diffusivity = diffusivity * np.array((1.0, 4.0, 2.0))

        def derivative(_time: float, flat_total: np.ndarray) -> np.ndarray:
            species = equilibrium(flat_total.reshape(nodes, 2))
            # Central interface values.  The potential gradient is chosen so
            # that Ja + Jb - Jc = 0 at each interface.
            face_concentration = 0.5 * (species[:-1] + species[1:])
            concentration_gradient = np.diff(species, axis=0) / dx
            current_gradient = (
                ion_diffusivity[0] * concentration_gradient[:, 0]
                + ion_diffusivity[1] * concentration_gradient[:, 1]
                - ion_diffusivity[2] * concentration_gradient[:, 2]
            )
            current_conductance = (
                ion_diffusivity[0] * face_concentration[:, 0]
                + ion_diffusivity[1] * face_concentration[:, 1]
                + ion_diffusivity[2] * face_concentration[:, 2]
            )
            # A face with no particles on either side carries neither charge
            # nor mass; its otherwise undefined electric field is immaterial.
            field_gradient = np.divide(
                -current_gradient,
                current_conductance,
                out=np.zeros_like(current_gradient),
                where=current_conductance > 0.0,
            )

            free_flux = -ion_diffusivity[:2] * (
                concentration_gradient[:, :2] + face_concentration[:, :2] * field_gradient[:, None]
            )
            cluster_flux = -diffusivity * NEUTRAL_MOBILITY * concentration_gradient[:, 3]
            total_flux = np.zeros((nodes + 1, 2))
            total_flux[1:-1] = free_flux + cluster_flux[:, None]
            return (-np.diff(total_flux, axis=0) / control_volume_width[:, None]).ravel()

        # Each two-component control volume depends only on itself and its
        # neighbours.  Supplying this banded structure keeps BDF inexpensive.
        entries = 2 * nodes
        offsets = np.arange(-3, 4)
        sparsity = diags([np.ones(entries - abs(offset)) for offset in offsets], offsets, format="csr")
        solution = solve_ivp(
            derivative,
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
        prediction = solution.y.T.reshape(t.size, nodes, 2)
        if not np.isfinite(prediction).all():
            raise RuntimeError("transport solver produced a non-finite concentration")
        return prediction


def fluctuation_matrices(totals, diffusivity):
    """Return local total susceptibility and constrained Onsager mobility."""
    totals = np.asarray(totals, dtype=float)
    if totals.shape != (2,) or not np.isfinite(totals).all() or np.any(totals <= 0.0):
        raise ValueError("totals must be a positive length-two array")
    if not np.isfinite(diffusivity) or diffusivity <= 0.0:
        raise ValueError("diffusivity must be positive")

    concentration = equilibrium(totals[None, :])[0]
    charge = np.array((1.0, 1.0, -1.0, 0.0))
    constituents = np.array(((1.0, 0.0, 0.0, 1.0), (0.0, 1.0, 0.0, 1.0)))

    # Projection removes the instantaneous charge degree of freedom.  The
    # first matrix is d(total)/d(mu_total); the second is its corresponding
    # zero-current Onsager mobility.
    population = np.diag(concentration)
    charge_population = population @ charge
    conditional_population = population - np.outer(charge_population, charge_population) / (
        charge @ charge_population
    )
    susceptibility = constituents @ conditional_population @ constituents.T

    species_mobility = np.diag(float(diffusivity) * np.array((1.0, 4.0, 2.0, NEUTRAL_MOBILITY)) * concentration)
    charge_mobility = species_mobility @ charge
    constrained_mobility = species_mobility - np.outer(charge_mobility, charge_mobility) / (
        charge @ charge_mobility
    )
    mobility = constituents @ constrained_mobility @ constituents.T
    return susceptibility, mobility


def equilibrium_correlations(t, totals, harmonic, diffusivity):
    """Two-time covariance of a nonzero cosine mode.

    The specified detector convention supplies independent ideal total-
    constituent fluctuations at zero lag, hence ``diag(totals)`` rather than
    a reaction-coordinate covariance is used as the right-hand factor.
    """
    t = _check_times(np.asarray(t, dtype=float))
    harmonic = int(harmonic)
    if harmonic < 1 or harmonic > 8:
        raise ValueError("harmonic must be an integer from 1 through 8")
    susceptibility, mobility = fluctuation_matrices(totals, diffusivity)
    relaxation = (harmonic * np.pi / CELL_LENGTH) ** 2 * np.linalg.solve(susceptibility.T, mobility.T).T
    initial_covariance = np.diag(np.asarray(totals, dtype=float))
    return np.stack([expm(-float(lag) * relaxation) @ initial_covariance for lag in t])


def correlate(self, t, totals, harmonic=1):
    """Return the detector covariance matrix for a cosine spatial mode."""
    return equilibrium_correlations(t, totals, harmonic, self._require_fit())


# Kept as a method assignment for compatibility with the original module's
# public layout, while retaining a separately testable implementation above.
TransportModel.correlate = correlate
