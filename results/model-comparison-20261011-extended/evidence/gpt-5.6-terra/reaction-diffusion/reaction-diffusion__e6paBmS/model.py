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
    """Return local free A, free B, free C, and cluster concentrations."""
    values = np.asarray(totals, dtype=float)
    if values.ndim == 0 or values.shape[-1] != 2:
        raise ValueError("totals must have final dimension 2 (A, B)")
    shape = values.shape[:-1]
    flat = values.reshape(-1, 2)
    a_total, b_total = flat.T
    if np.any(a_total < 0) or np.any(b_total < 0):
        raise ValueError("total concentrations must be nonnegative")

    # p = K(a-p)(b-p)(a+b-2p)^2 is monotone on [0,min(a,b)].
    p = np.zeros_like(a_total)
    upper = np.minimum(a_total, b_total)
    for _ in range(32):
        a, b = a_total - p, b_total - p
        c = a + b
        residual = p - ASSOCIATION * a * b * c**2
        derivative = 1.0 + ASSOCIATION * (c**3 + 4.0 * a * b * c)
        updated = np.clip(p - residual / derivative, 0.0, upper)
        if np.max(np.abs(updated - p), initial=0.0) < 2e-14:
            p = updated
            break
        p = updated
    a, b = a_total - p, b_total - p
    return np.column_stack((a, b, a + b, p)).reshape(shape + (4,))


def _validate_grid(t, x, initial):
    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)
    initial = np.asarray(initial, dtype=float)
    if t.ndim != 1 or not t.size or np.any(t < 0) or np.any(np.diff(t) < 0):
        raise ValueError("t must be a nonempty increasing array of nonnegative times")
    if x.ndim != 1 or x.size < 2 or np.any(np.diff(x) <= 0):
        raise ValueError("x must be an increasing one-dimensional mesh")
    dx = x[1] - x[0]
    if not np.allclose(np.diff(x), dx, rtol=2e-10, atol=1e-14):
        raise ValueError("x must be equally spaced")
    if initial.shape != (x.size, 2):
        raise ValueError("initial must have shape (len(x), 2)")
    if not np.isfinite(initial).all() or np.any(initial < 0):
        raise ValueError("initial concentrations must be finite and nonnegative")
    return t, x, initial


class TransportModel:
    def __init__(self):
        self.diffusivity = None

    @staticmethod
    def _binary_with_diffusivity(t, x, initial, diffusivity):
        """Exact-in-time discrete Neumann solution for a one-cation salt."""
        intervals = len(x) - 1
        dx = x[1] - x[0]
        eig = 4.0 / dx**2 * np.sin(np.arange(intervals + 1) * np.pi / (2.0 * intervals)) ** 2
        modes = dct(initial, type=1, axis=0)
        # 2 D+ D-/(D+ + D-) for A-C and B-C, respectively.
        rates = eig[:, None] * diffusivity * np.array([4.0 / 3.0, 8.0 / 3.0])
        return idct(np.exp(-t[:, None, None] * rates[None]) * modes[None], type=1, axis=1)

    def _binary(self, t, x, initial):
        return self._binary_with_diffusivity(t, x, initial, self.diffusivity)

    def fit(self, data):
        """Fit the common ionic diffusion scale by weighted least squares."""
        try:
            t = np.asarray(data["t"], dtype=float)
            x = np.asarray(data["x"], dtype=float)
            initials = np.asarray(data["initial"], dtype=float)
            observed = np.asarray(data["concentration"], dtype=float)
            sigma = np.asarray(data["sigma"], dtype=float)
        except (KeyError, TypeError) as exc:
            raise ValueError("data must contain x, t, initial, concentration, and sigma") from exc
        if initials.ndim != 3 or initials.shape[2] != 2:
            raise ValueError("initial must have shape (records, x, 2)")
        if observed.shape != (initials.shape[0], t.size, x.size, 2) or sigma.shape != observed.shape:
            raise ValueError("calibration array shapes are inconsistent")
        if not np.isfinite(observed).all() or np.any(~np.isfinite(sigma)) or np.any(sigma <= 0):
            raise ValueError("calibration values must be finite and sigma positive")
        for initial in initials:
            _validate_grid(t, x, initial)
        # Calibration preparations are deliberately cluster-free.  This gives
        # an exact, fast likelihood evaluation at each trial diffusivity.
        if not all(np.all(item[:, 0] == 0.0) or np.all(item[:, 1] == 0.0) for item in initials):
            raise ValueError("calibration preparations must contain one cation constituent")

        def objective(log_d):
            d = float(np.exp(log_d))
            return sum(
                float(np.sum(((self._binary_with_diffusivity(t, x, initial, d) - measurement) / uncertainty) ** 2))
                for initial, measurement, uncertainty in zip(initials, observed, sigma)
            )

        outcome = minimize_scalar(
            objective, bounds=(np.log(1e-13), np.log(1e-6)), method="bounded", options={"xatol": 2e-11}
        )
        if not outcome.success or not np.isfinite(outcome.fun):
            raise RuntimeError("diffusivity fit did not converge")
        self.diffusivity = float(np.exp(outcome.x))
        return self

    def predict(self, t, x, initial):
        if self.diffusivity is None or not np.isfinite(self.diffusivity) or self.diffusivity <= 0:
            raise RuntimeError("fit must be called before predict")
        t, x, initial = _validate_grid(t, x, initial)
        if np.all(initial[:, 0] == 0.0) or np.all(initial[:, 1] == 0.0):
            return self._binary(t, x, initial)
        if t[-1] == 0.0:
            return np.repeat(initial[None, :, :], t.size, axis=0)

        count, dx = x.size, x[1] - x[0]
        widths = np.full(count, dx)
        widths[[0, -1]] *= 0.5
        d_a, d_b, d_c = self.diffusivity * np.array([1.0, 4.0, 2.0])

        def rhs(_, packed):
            # Solver trial points can have tiny negative values; treating them
            # as zero avoids asking the chemical equilibrium to extrapolate.
            species = equilibrium(np.maximum(packed.reshape(count, 2), 0.0))
            face = 0.5 * (species[:-1] + species[1:])
            gradient = np.diff(species, axis=0) / dx
            phi_x = -(d_a * gradient[:, 0] + d_b * gradient[:, 1] - d_c * gradient[:, 2]) / (
                d_a * face[:, 0] + d_b * face[:, 1] + d_c * face[:, 2]
            )
            free_flux = -np.array([d_a, d_b]) * (gradient[:, :2] + face[:, :2] * phi_x[:, None])
            cluster_flux = -self.diffusivity * NEUTRAL_MOBILITY * gradient[:, 3]
            flux = np.zeros((count + 1, 2))  # blocking boundaries
            flux[1:-1] = free_flux + cluster_flux[:, None]
            return (-np.diff(flux, axis=0) / widths[:, None]).ravel()

        # Two state variables per node; only adjacent nodes interact.
        bandwidth = 3
        sparsity = diags(
            [np.ones(2 * count - abs(k)) for k in range(-bandwidth, bandwidth + 1)],
            range(-bandwidth, bandwidth + 1), shape=(2 * count, 2 * count), format="csr"
        )
        solved = solve_ivp(
            rhs, (0.0, float(t[-1])), initial.ravel(), method="BDF", t_eval=t,
            rtol=2e-8, atol=2e-10, jac_sparsity=sparsity,
        )
        if not solved.success:
            raise RuntimeError(solved.message)
        return solved.y.T.reshape(t.size, count, 2)

    def correlate(self, t, totals, harmonic=1):
        """Covariance of a nonzero equilibrium cosine mode, in A,B order."""
        if self.diffusivity is None or not np.isfinite(self.diffusivity) or self.diffusivity <= 0:
            raise RuntimeError("fit must be called before correlate")
        times, total = np.asarray(t, dtype=float), np.asarray(totals, dtype=float)
        if times.ndim != 1 or np.any(~np.isfinite(times)) or np.any(times < 0) or np.any(np.diff(times) < 0):
            raise ValueError("t must be an increasing one-dimensional array of nonnegative times")
        if total.shape != (2,) or np.any(~np.isfinite(total)) or np.any(total <= 0):
            raise ValueError("totals must be two positive concentrations")
        if int(harmonic) != harmonic or not 1 <= int(harmonic) <= 8:
            raise ValueError("harmonic must be an integer from 1 through 8")
        if times.size and times[-1] > 600.0:
            raise ValueError("correlation lags must not exceed 600 s")

        susceptibility, mobility = fluctuation_matrices(total, self.diffusivity)
        relaxation = (int(harmonic) * np.pi / CELL_LENGTH) ** 2 * mobility @ np.linalg.inv(susceptibility)
        equal_time = np.diag(total)  # specified independent ideal total fluctuations
        return np.array([expm(-float(lag) * relaxation) @ equal_time for lag in times])


def fluctuation_matrices(totals, diffusivity):
    """Return the charge-constrained susceptibility and Onsager mobility."""
    species = equilibrium(np.asarray(totals, dtype=float))
    if species.shape != (4,):
        raise ValueError("totals must have shape (2,)")
    charge = np.array([1.0, 1.0, -1.0, 0.0])
    constituents = np.array([[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0]])
    population = np.diag(species)
    constrained_population = population - np.outer(population @ charge, population @ charge) / (charge @ population @ charge)
    susceptibility = constituents @ constrained_population @ constituents.T

    particle_mobility = np.diag(diffusivity * np.array([1.0, 4.0, 2.0, NEUTRAL_MOBILITY]) * species)
    charged_mobility = particle_mobility @ charge
    constrained_mobility = particle_mobility - np.outer(charged_mobility, charged_mobility) / (
        charge @ particle_mobility @ charge
    )
    mobility = constituents @ constrained_mobility @ constituents.T
    return susceptibility, mobility


def equilibrium_correlations(t, totals, harmonic, diffusivity):
    """Backwards-compatible module-level correlation helper."""
    times = np.asarray(t, dtype=float)
    susceptibility, mobility = fluctuation_matrices(totals, diffusivity)
    relaxation = (int(harmonic) * np.pi / CELL_LENGTH) ** 2 * mobility @ np.linalg.inv(susceptibility)
    return np.array([expm(-float(lag) * relaxation) @ np.diag(totals) for lag in times])
