"""Transport model for the reversible electrolyte cell.

The public state is the pair of *total* A and B constituent concentrations.
Free concentrations are recovered locally from mass action for each flux.
"""

import numpy as np
from scipy.fft import dct, idct
from scipy.integrate import solve_ivp
from scipy.linalg import expm
from scipy.optimize import minimize_scalar
from scipy.sparse import diags


ASSOCIATION = 0.5
NEUTRAL_MOBILITY = 0.15
CELL_LENGTH = 1.0e-3


def equilibrium(total):
    """Return free A, free B, free C, and P for total A/B concentrations."""
    total = np.asarray(total, dtype=float)
    if total.shape[-1] != 2:
        raise ValueError("total concentrations must have a final dimension of two")
    a, b = total[..., 0], total[..., 1]
    if np.any(a < 0) or np.any(b < 0):
        raise ValueError("total concentrations must be non-negative")
    p = np.zeros(np.broadcast(a, b).shape, dtype=float)
    upper = np.minimum(a, b)
    for _ in range(40):
        af, bf = a - p, b - p
        cf = af + bf
        value = p - ASSOCIATION * af * bf * cf**2
        derivative = 1.0 + ASSOCIATION * (cf**3 + 4.0 * af * bf * cf)
        candidate = np.clip(p - value / derivative, 0.0, upper)
        if np.max(np.abs(candidate - p), initial=0.0) < 2e-14:
            p = candidate
            break
        p = candidate
    return np.stack((a - p, b - p, a + b - 2.0 * p, p), axis=-1)


def _binary_profile(t, x, initial, diffusivity):
    """Exact discrete Neumann solution for a preparation containing one salt."""
    n = len(x) - 1
    dx = x[1] - x[0]
    eigenvalue = 4.0 / dx**2 * np.sin(np.arange(n + 1) * np.pi / (2.0 * n)) ** 2
    coefficient = dct(initial, type=1, axis=0)
    rate = eigenvalue[:, None] * diffusivity * np.array((4.0 / 3.0, 8.0 / 3.0))
    factor = np.exp(-np.asarray(t)[:, None, None] * rate[None, :, :])
    return idct(factor * coefficient[None, :, :], type=1, axis=1)


class TransportModel:
    def __init__(self):
        self.diffusivity = None

    def fit(self, data):
        """Fit the one positive diffusion scale by weighted least squares."""
        try:
            t = np.asarray(data["t"], dtype=float)
            x = np.asarray(data["x"], dtype=float)
            initial = np.asarray(data["initial"], dtype=float)
            observed = np.asarray(data["concentration"], dtype=float)
            sigma = np.asarray(data["sigma"], dtype=float)
        except KeyError as exc:
            raise ValueError(f"missing calibration field {exc.args[0]!r}") from exc
        if initial.ndim != 3 or initial.shape[-1] != 2:
            raise ValueError("initial must have shape (records, x, 2)")
        if observed.shape != (len(initial), len(t), len(x), 2) or sigma.shape != observed.shape:
            raise ValueError("inconsistent calibration array shapes")
        if np.any(sigma <= 0) or not np.isfinite(sigma).all():
            raise ValueError("sigma must be finite and positive")
        self._check_mesh(t, x)

        # Every calibration record is a binary salt preparation.  Its
        # zero-current ambipolar solution is exact, avoiding a stiff solve in
        # the one-dimensional fit.
        def objective(log_d):
            calculated = np.array([_binary_profile(t, x, item, np.exp(log_d)) for item in initial])
            residual = (calculated - observed) / sigma
            return float(np.sum(residual * residual))

        result = minimize_scalar(
            objective, bounds=(np.log(1e-14), np.log(1e-5)), method="bounded",
            options={"xatol": 2e-11},
        )
        if not result.success or not np.isfinite(result.fun):
            raise RuntimeError("could not fit diffusivity")
        self.diffusivity = float(np.exp(result.x))
        return self

    @staticmethod
    def _check_mesh(t, x):
        if t.ndim != 1 or x.ndim != 1 or len(t) == 0 or len(x) < 2:
            raise ValueError("t must be nonempty and x must contain at least two points")
        if not np.isfinite(t).all() or not np.isfinite(x).all() or np.any(t < 0):
            raise ValueError("t and x must be finite, with non-negative t")
        if len(t) > 1 and np.any(np.diff(t) <= 0):
            raise ValueError("t must be strictly increasing")
        dx = np.diff(x)
        if np.any(dx <= 0) or not np.allclose(dx, dx[0], rtol=2e-11, atol=1e-15):
            raise ValueError("x must be an equally spaced increasing mesh")

    def predict(self, t, x, initial):
        """Predict total A and B profiles at the requested preparation ages."""
        if self.diffusivity is None:
            raise RuntimeError("fit must be called before predict")
        t, x, initial = np.asarray(t, dtype=float), np.asarray(x, dtype=float), np.asarray(initial, dtype=float)
        self._check_mesh(t, x)
        if initial.shape != (len(x), 2):
            raise ValueError("initial must have shape (len(x), 2)")
        if np.any(initial < 0) or not np.isfinite(initial).all():
            raise ValueError("initial concentrations must be finite and non-negative")
        # With non-negative increasing times, this is the sole requested age
        # when the last entry is zero.  solve_ivp has no useful zero-length
        # integration result to reshape in that case.
        if t[-1] == 0.0:
            return initial[None, :, :].copy()
        if np.all(initial[:, 0] == 0.0) or np.all(initial[:, 1] == 0.0):
            return _binary_profile(t, x, initial, self.diffusivity)

        n, dx = len(x), x[1] - x[0]
        volume = np.full(n, dx)
        volume[[0, -1]] *= 0.5
        da, db, dc = self.diffusivity * np.array((1.0, 4.0, 2.0))

        def rhs(_time, state):
            species = equilibrium(state.reshape(n, 2))
            face = 0.5 * (species[:-1] + species[1:])
            gradient = np.diff(species, axis=0) / dx
            # Dimensionless electric gradient from vanishing free-ion current.
            electric = -(da * gradient[:, 0] + db * gradient[:, 1] - dc * gradient[:, 2])
            electric /= da * face[:, 0] + db * face[:, 1] + dc * face[:, 2]
            ion_flux = -np.column_stack((
                da * (gradient[:, 0] + face[:, 0] * electric),
                db * (gradient[:, 1] + face[:, 1] * electric),
            ))
            cluster_flux = -self.diffusivity * NEUTRAL_MOBILITY * gradient[:, 3]
            flux = np.zeros((n + 1, 2))  # blocking boundaries
            flux[1:-1] = ion_flux + cluster_flux[:, None]
            return (-np.diff(flux, axis=0) / volume[:, None]).ravel()

        sparsity = diags(
            [np.ones(2 * n - abs(k)) for k in range(-3, 4)], range(-3, 4),
            shape=(2 * n, 2 * n), format="csr",
        )
        solution = solve_ivp(
            rhs, (0.0, float(t[-1])), initial.ravel(), method="BDF", t_eval=t,
            rtol=2e-8, atol=2e-10, jac_sparsity=sparsity,
        )
        if not solution.success:
            raise RuntimeError(solution.message)
        answer = solution.y.T.reshape(len(t), n, 2)
        if not np.isfinite(answer).all():
            raise RuntimeError("non-finite concentration prediction")
        return answer

    def correlate(self, t, totals, harmonic=1):
        """Equilibrium cosine-mode covariance of total A and B constituents."""
        if self.diffusivity is None:
            raise RuntimeError("fit must be called before correlate")
        t, totals = np.asarray(t, dtype=float), np.asarray(totals, dtype=float)
        if t.ndim != 1 or len(t) == 0 or np.any(t < 0) or not np.isfinite(t).all():
            raise ValueError("t must be a nonempty finite non-negative array")
        if len(t) > 1 and np.any(np.diff(t) <= 0):
            raise ValueError("t must be strictly increasing")
        if totals.shape != (2,) or np.any(totals <= 0) or not np.isfinite(totals).all():
            raise ValueError("totals must contain two finite positive concentrations")
        if isinstance(harmonic, (bool, np.bool_)) or int(harmonic) != harmonic or not 1 <= int(harmonic) <= 8:
            raise ValueError("harmonic must be an integer from 1 through 8")
        susceptibility, mobility = fluctuation_matrices(totals, self.diffusivity)
        wave_number = int(harmonic) * np.pi / CELL_LENGTH
        relaxation = wave_number**2 * mobility @ np.linalg.inv(susceptibility)
        # The detector convention specifies independent ideal total-constituent
        # fluctuations at zero lag.
        return np.array([expm(-float(lag) * relaxation) @ np.diag(totals) for lag in t])


def fluctuation_matrices(totals, diffusivity):
    """Susceptibility and Onsager mobility after charge/reaction elimination."""
    c = equilibrium(np.asarray(totals, dtype=float)[None, :])[0]
    charge = np.array((1.0, 1.0, -1.0, 0.0))
    constituent = np.array(((1.0, 0.0, 0.0, 1.0), (0.0, 1.0, 0.0, 1.0)))
    population = np.diag(c)
    conditional = population - np.outer(population @ charge, population @ charge) / (charge @ population @ charge)
    susceptibility = constituent @ conditional @ constituent.T
    species_mobility = np.diag(diffusivity * np.array((1.0, 4.0, 2.0, NEUTRAL_MOBILITY)) * c)
    mobile_charge = species_mobility @ charge
    constrained = species_mobility - np.outer(mobile_charge, mobile_charge) / (charge @ species_mobility @ charge)
    mobility = constituent @ constrained @ constituent.T
    return susceptibility, mobility


def equilibrium_correlations(t, totals, harmonic, diffusivity):
    """Compatibility helper retained from the original module API."""
    susceptibility, mobility = fluctuation_matrices(totals, diffusivity)
    rate = (int(harmonic) * np.pi / CELL_LENGTH) ** 2 * mobility @ np.linalg.inv(susceptibility)
    return np.array([expm(-float(time) * rate) @ np.diag(totals) for time in t])
