"""Electroneutral transport model for the electrolyte cell."""

import numpy as np
from scipy.fft import dct, idct
from scipy.integrate import solve_ivp
from scipy.linalg import expm
from scipy.optimize import minimize_scalar
from scipy.sparse import diags


ASSOCIATION = 0.5
NEUTRAL_MOBILITY = 0.15
CELL_LENGTH = 1.0e-3


def equilibrium(totals):
    """Return free A, free B, free C, and P at local mass-action equilibrium."""
    totals = np.asarray(totals, dtype=float)
    if totals.shape[-1] != 2:
        raise ValueError("totals must have final dimension 2")
    if not np.all(np.isfinite(totals)) or np.any(totals < 0):
        raise ValueError("total concentrations must be finite and nonnegative")

    a, b = totals[..., 0], totals[..., 1]
    upper = np.minimum(a, b)
    lo, hi = np.zeros_like(upper), upper.copy()
    p = 0.5 * upper
    # f(P)=P-K(A-P)(B-P)(A+B-2P)^2 is strictly increasing on its
    # physical interval.  Safeguarding Newton with that bracket is robust
    # at pure preparations as well as at strongly bound mixtures.
    for _ in range(48):
        fa, fb = a - p, b - p
        c = fa + fb
        f = p - ASSOCIATION * fa * fb * c**2
        fp = 1.0 + ASSOCIATION * (c**3 + 4.0 * fa * fb * c)
        lo = np.where(f < 0.0, p, lo)
        hi = np.where(f >= 0.0, p, hi)
        trial = p - f / fp
        p = np.where((trial > lo) & (trial < hi) & np.isfinite(trial),
                     trial, 0.5 * (lo + hi))
        if np.max(hi - lo, initial=0.0) < 2e-14:
            break
    fa, fb = a - p, b - p
    c = fa + fb
    p = np.clip(p - (p - ASSOCIATION * fa * fb * c**2) /
                (1.0 + ASSOCIATION * (c**3 + 4.0 * fa * fb * c)), 0.0, upper)
    return np.stack((a - p, b - p, a + b - 2.0 * p, p), axis=-1)


def _check_grid(t, x, initial):
    t, x, initial = (np.asarray(t, dtype=float), np.asarray(x, dtype=float),
                     np.asarray(initial, dtype=float))
    if t.ndim != 1 or x.ndim != 1 or initial.shape != (len(x), 2):
        raise ValueError("expected t=(T,), x=(X,), and initial=(X,2)")
    if len(x) < 2 or not np.all(np.isfinite(x)) or not np.all(np.isfinite(t)):
        raise ValueError("t and x must be finite, with at least two x points")
    if np.any(t < 0.0) or np.any(np.diff(t) <= 0.0):
        raise ValueError("t must be an increasing nonnegative array")
    dx = np.diff(x)
    if np.any(dx <= 0.0) or not np.allclose(dx, dx[0], rtol=2e-12, atol=1e-15):
        raise ValueError("x must be an equally spaced increasing mesh")
    if not np.all(np.isfinite(initial)) or np.any(initial < 0.0):
        raise ValueError("initial concentrations must be finite and nonnegative")
    return t, x, initial


class TransportModel:
    def __init__(self):
        self.diffusivity = None

    def fit(self, data):
        """Fit the common ionic diffusion coefficient by weighted least squares."""
        try:
            t = np.asarray(data["t"], dtype=float)
            x = np.asarray(data["x"], dtype=float)
            starts = np.asarray(data["initial"], dtype=float)
            observed = np.asarray(data["concentration"], dtype=float)
            sigma = np.asarray(data["sigma"], dtype=float)
        except (KeyError, TypeError) as exc:
            raise ValueError("data must contain x, t, initial, concentration, and sigma") from exc
        if starts.ndim != 3 or starts.shape[1:] != (len(x), 2):
            raise ValueError("initial must have shape (records, len(x), 2)")
        if observed.shape != (len(starts), len(t), len(x), 2) or sigma.shape != observed.shape:
            raise ValueError("concentration and sigma have incompatible shapes")
        if not np.all(np.isfinite(observed)) or not np.all(np.isfinite(sigma)) or np.any(sigma <= 0):
            raise ValueError("observations must be finite and sigma must be positive")
        _check_grid(t, x, starts[0])

        def objective(log_d):
            self.diffusivity = float(np.exp(log_d))
            try:
                predicted = np.stack([self.predict(t, x, start) for start in starts])
            except (ValueError, RuntimeError, FloatingPointError):
                return np.inf
            residual = (predicted - observed) / sigma
            value = float(np.sum(residual * residual))
            return value if np.isfinite(value) else np.inf

        result = minimize_scalar(
            objective, bounds=(np.log(1e-14), np.log(1e-6)), method="bounded",
            options={"xatol": 2e-11, "maxiter": 100},
        )
        if not result.success or not np.isfinite(result.fun):
            self.diffusivity = None
            raise RuntimeError("could not fit diffusivity")
        self.diffusivity = float(np.exp(result.x))
        return self

    def _require_diffusivity(self):
        if self.diffusivity is None or not np.isfinite(self.diffusivity) or self.diffusivity <= 0:
            raise RuntimeError("fit must be called before prediction")

    def _binary(self, t, x, initial):
        """Exact semi-discrete Neumann solution for a pure-cation preparation."""
        n = len(x) - 1
        dx = x[1] - x[0]
        eigenvalue = 4.0 / dx**2 * np.sin(np.arange(n + 1) * np.pi / (2.0 * n))**2
        coefficients = dct(initial, type=1, axis=0)
        # A+/C- and B+/C- ambipolar diffusivities are 4D/3 and 8D/3.
        effective = self.diffusivity * np.array((4.0 / 3.0, 8.0 / 3.0))
        multiplier = np.exp(-t[:, None, None] * eigenvalue[None, :, None]
                            * effective[None, None, :])
        return idct(multiplier * coefficients[None, :, :], type=1, axis=1)

    def predict(self, t, x, initial):
        self._require_diffusivity()
        t, x, initial = _check_grid(t, x, initial)
        if len(t) == 0:
            return np.empty((0, len(x), 2), dtype=float)
        if t[-1] == 0.0:
            return np.broadcast_to(initial, (len(t),) + initial.shape).copy()
        # These are precisely the calibration preparations; no clusters form.
        if np.all(initial[:, 0] == 0.0) or np.all(initial[:, 1] == 0.0):
            return self._binary(t, x, initial)

        points, dx = len(x), x[1] - x[0]
        volumes = np.full(points, dx)
        volumes[[0, -1]] *= 0.5
        d_a, d_b, d_c = self.diffusivity * np.array((1.0, 4.0, 2.0))

        def rhs(_time, flattened):
            species = equilibrium(flattened.reshape(points, 2))
            face = 0.5 * (species[:-1] + species[1:])
            gradient = np.diff(species, axis=0) / dx
            # Select the electric field which cancels charge current at every
            # face.  The neutral P flux carries no charge.
            numerator = -(d_a * gradient[:, 0] + d_b * gradient[:, 1]
                          - d_c * gradient[:, 2])
            denominator = d_a * face[:, 0] + d_b * face[:, 1] + d_c * face[:, 2]
            # A completely empty face has both a zero numerator and no
            # physically meaningful field; assigning zero avoids 0/0.
            phi_x = np.divide(numerator, denominator, out=np.zeros_like(numerator),
                              where=denominator > 0.0)
            ion_flux = -np.array((d_a, d_b)) * (
                gradient[:, :2] + face[:, :2] * phi_x[:, None])
            p_flux = -self.diffusivity * NEUTRAL_MOBILITY * gradient[:, 3]
            flux = np.zeros((points + 1, 2))
            flux[1:-1] = ion_flux + p_flux[:, None]
            return (-np.diff(flux, axis=0) / volumes[:, None]).ravel()

        # A node depends on at most its two neighbours (two state values each).
        state_size = 2 * points
        pattern = diags([np.ones(state_size - abs(k)) for k in range(-3, 4)],
                        range(-3, 4), shape=(state_size, state_size), format="csr")
        solution = solve_ivp(rhs, (0.0, float(t[-1])), initial.ravel(), method="BDF",
                             t_eval=t, rtol=2e-8, atol=2e-10, jac_sparsity=pattern)
        if not solution.success:
            raise RuntimeError(solution.message)
        answer = solution.y.T.reshape(len(t), points, 2)
        if not np.all(np.isfinite(answer)):
            raise RuntimeError("transport integration produced non-finite concentrations")
        return answer

    def correlate(self, t, totals, harmonic=1):
        """Return the documented nonzero-mode equilibrium covariance."""
        self._require_diffusivity()
        t, totals = np.asarray(t, dtype=float), np.asarray(totals, dtype=float)
        if (t.ndim != 1 or not np.all(np.isfinite(t)) or np.any(t < 0)
                or np.any(np.diff(t) <= 0)):
            raise ValueError("t must be an increasing nonnegative array")
        if totals.shape != (2,) or not np.all(np.isfinite(totals)) or np.any(totals <= 0):
            raise ValueError("totals must be two positive concentrations")
        if isinstance(harmonic, bool) or int(harmonic) != harmonic or not 1 <= int(harmonic) <= 8:
            raise ValueError("harmonic must be an integer from 1 through 8")
        if len(t) == 0:
            return np.empty((0, 2, 2), dtype=float)
        susceptibility, mobility = fluctuation_matrices(totals, self.diffusivity)
        q = int(harmonic) * np.pi / CELL_LENGTH
        relaxation = q**2 * np.linalg.solve(susceptibility.T, mobility.T).T
        # This is specified as independent ideal total-constituent noise.
        equal_time = np.diag(totals)
        return np.stack([expm(-lag * relaxation) @ equal_time for lag in t])


def fluctuation_matrices(totals, diffusivity):
    """Return total susceptibility and zero-current Onsager mobility."""
    concentration = equilibrium(np.asarray(totals, dtype=float))
    if concentration.shape != (4,):
        raise ValueError("totals must have shape (2,)")
    charge = np.array((1.0, 1.0, -1.0, 0.0))
    constituents = np.array(((1.0, 0.0, 0.0, 1.0),
                             (0.0, 1.0, 0.0, 1.0)))
    population = np.diag(concentration)
    conditional = population - np.outer(population @ charge, population @ charge) / (
        charge @ population @ charge)
    susceptibility = constituents @ conditional @ constituents.T

    species_mobility = np.diag(
        diffusivity * np.array((1.0, 4.0, 2.0, NEUTRAL_MOBILITY)) * concentration)
    charge_mobility = species_mobility @ charge
    constrained = species_mobility - np.outer(charge_mobility, charge_mobility) / (
        charge @ species_mobility @ charge)
    return susceptibility, constituents @ constrained @ constituents.T


def equilibrium_correlations(t, totals, harmonic, diffusivity):
    """Compatibility helper for callers of the previous module API."""
    susceptibility, mobility = fluctuation_matrices(totals, diffusivity)
    relaxation = (int(harmonic) * np.pi / CELL_LENGTH)**2 * np.linalg.solve(
        susceptibility.T, mobility.T).T
    return np.stack([expm(-float(lag) * relaxation) @ np.diag(totals) for lag in t])
