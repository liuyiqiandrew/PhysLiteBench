"""Bulk electroneutral transport model for the association cell."""

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
        raise ValueError("total concentrations must have final dimension 2")
    a_total, b_total = total[..., 0], total[..., 1]
    if np.any(a_total < 0) or np.any(b_total < 0):
        raise ValueError("total concentrations must be nonnegative")

    # The mass-action equation has one physical root in [0, min(A, B)].
    bound = np.zeros_like(a_total)
    upper = np.minimum(a_total, b_total)
    for _ in range(48):
        a, b = a_total - bound, b_total - bound
        c = a + b
        residual = bound - ASSOCIATION * a * b * c**2
        slope = 1.0 + ASSOCIATION * (c**3 + 4.0 * a * b * c)
        candidate = np.clip(bound - residual / slope, 0.0, upper)
        if np.max(np.abs(candidate - bound), initial=0.0) < 2e-14:
            bound = candidate
            break
        bound = candidate
    a, b = a_total - bound, b_total - bound
    return np.stack((a, b, a + b, bound), axis=-1)


class TransportModel:
    def __init__(self):
        self.diffusivity = None

    def fit(self, data):
        """Fit the common ionic diffusion scale by weighted least squares."""
        x = np.asarray(data["x"], dtype=float)
        times = np.asarray(data["t"], dtype=float)
        initial = np.asarray(data["initial"], dtype=float)
        observed = np.asarray(data["concentration"], dtype=float)
        sigma = np.asarray(data["sigma"], dtype=float)
        if initial.ndim != 3 or initial.shape[2] != 2:
            raise ValueError("initial must have shape (record, x, 2)")
        if observed.shape != (len(initial), len(times), len(x), 2):
            raise ValueError("concentration has an incompatible shape")
        if sigma.shape != observed.shape or np.any(sigma <= 0):
            raise ValueError("sigma must be positive and match concentration")

        # The logarithmic parameterization makes positivity exact.
        def objective(log_diffusivity):
            self.diffusivity = float(np.exp(log_diffusivity))
            predicted = np.empty_like(observed)
            for record, profile in enumerate(initial):
                predicted[record] = self.predict(times, x, profile)
            residual = (predicted - observed) / sigma
            return float(np.sum(residual * residual))

        result = minimize_scalar(
            objective, bounds=(np.log(1e-13), np.log(1e-6)), method="bounded",
            options={"xatol": 1e-11},
        )
        if not result.success or not np.isfinite(result.fun):
            raise RuntimeError("diffusivity fit did not converge")
        self.diffusivity = float(np.exp(result.x))
        return self

    def _binary(self, t, x, initial):
        """Exact semi-discrete no-flux solution for an unmixed preparation."""
        intervals = len(x) - 1
        dx = x[1] - x[0]
        eigenvalue = 4.0 / dx**2 * np.sin(
            np.arange(intervals + 1) * np.pi / (2.0 * intervals)
        ) ** 2
        coefficients = dct(initial, type=1, axis=0)
        # 2 D+ D-/(D+ + D-) for A/C and B/C, respectively.
        ambipolar = self.diffusivity * np.array((4.0 / 3.0, 8.0 / 3.0))
        attenuation = np.exp(-t[:, None, None] * eigenvalue[None, :, None]
                             * ambipolar[None, None, :])
        return idct(attenuation * coefficients[None], type=1, axis=1)

    def predict(self, t, x, initial):
        """Predict total A and B profiles at the requested ages."""
        if self.diffusivity is None or not np.isfinite(self.diffusivity) or self.diffusivity <= 0:
            raise RuntimeError("fit the model before predicting")
        t, x, initial = np.asarray(t, dtype=float), np.asarray(x, dtype=float), np.asarray(initial, dtype=float)
        if t.ndim != 1 or len(t) == 0 or np.any(t < 0) or np.any(np.diff(t) < 0):
            raise ValueError("t must be a nonempty increasing nonnegative array")
        if x.ndim != 1 or len(x) < 2 or initial.shape != (len(x), 2):
            raise ValueError("initial must have shape (len(x), 2)")
        dx = x[1] - x[0]
        if dx <= 0 or not np.allclose(np.diff(x), dx, rtol=2e-12, atol=1e-15):
            raise ValueError("x must be an equally spaced increasing mesh")
        if np.any(initial < 0):
            raise ValueError("initial concentrations must be nonnegative")

        # Avoid invoking an ODE solver on its degenerate [0, 0] interval.
        if t[-1] == 0.0:
            return initial[None].copy()

        # Association vanishes identically for all calibration preparations.
        if np.all(initial[:, 0] == 0.0) or np.all(initial[:, 1] == 0.0):
            return self._binary(t, x, initial)

        points = len(x)
        volume = np.full(points, dx)
        volume[[0, -1]] *= 0.5
        d_a, d_b, d_c = self.diffusivity * np.array((1.0, 4.0, 2.0))

        def rhs(_, flat):
            species = equilibrium(flat.reshape(points, 2))
            midpoint = 0.5 * (species[:-1] + species[1:])
            gradient = np.diff(species, axis=0) / dx
            # Enforce J_A + J_B - J_C = 0 on each interior face.
            numerator = d_a * gradient[:, 0] + d_b * gradient[:, 1] - d_c * gradient[:, 2]
            denominator = d_a * midpoint[:, 0] + d_b * midpoint[:, 1] + d_c * midpoint[:, 2]
            potential_slope = np.divide(-numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0)
            free_flux = -np.array((d_a, d_b)) * (gradient[:, :2] + midpoint[:, :2] * potential_slope[:, None])
            cluster_flux = -self.diffusivity * NEUTRAL_MOBILITY * gradient[:, 3]
            total_flux = np.zeros((points + 1, 2))  # blocked ends
            total_flux[1:-1] = free_flux + cluster_flux[:, None]
            return (-np.diff(total_flux, axis=0) / volume[:, None]).ravel()

        sparsity = diags([np.ones(2 * points - abs(offset)) for offset in range(-3, 4)],
                         range(-3, 4), shape=(2 * points, 2 * points), format="csr")
        solved = solve_ivp(rhs, (0.0, float(t[-1])), initial.ravel(), method="BDF", t_eval=t,
                           rtol=2e-8, atol=2e-10, jac_sparsity=sparsity)
        if not solved.success:
            raise RuntimeError(solved.message)
        return np.maximum(solved.y.T.reshape(len(t), points, 2), 0.0)

    def correlate(self, t, totals, harmonic=1):
        """Return normalized two-time equilibrium A/B covariance matrices."""
        if self.diffusivity is None or not np.isfinite(self.diffusivity) or self.diffusivity <= 0:
            raise RuntimeError("fit the model before correlating")
        t, totals = np.asarray(t, dtype=float), np.asarray(totals, dtype=float)
        if t.ndim != 1 or len(t) == 0 or np.any(t < 0) or np.any(np.diff(t) < 0):
            raise ValueError("t must be a nonempty increasing nonnegative array")
        if totals.shape != (2,) or np.any(totals <= 0):
            raise ValueError("totals must be two positive concentrations")
        if int(harmonic) != harmonic or not 1 <= int(harmonic) <= 8:
            raise ValueError("harmonic must be an integer from 1 through 8")

        susceptibility, mobility = fluctuation_matrices(totals, self.diffusivity)
        wavenumber = int(harmonic) * np.pi / CELL_LENGTH
        relaxation = wavenumber**2 * np.linalg.solve(susceptibility, mobility.T).T
        return np.array([expm(-float(lag) * relaxation) @ susceptibility for lag in t])


def fluctuation_matrices(totals, diffusivity):
    """Return the fast-equilibrium susceptibility and Onsager mobility.

    Ideal species have covariance diag(c) and flux mobility diag(D_i c_i).
    Conditioning both on zero charge current/charge, then projecting to A/B
    constituent totals, accounts for both electroneutrality and association.
    """
    concentration = equilibrium(np.asarray(totals, dtype=float))
    charge = np.array((1.0, 1.0, -1.0, 0.0))
    constituent = np.array(((1.0, 0.0, 0.0, 1.0),
                            (0.0, 1.0, 0.0, 1.0)))
    population = np.diag(concentration)
    susceptibility_species = population - np.outer(population @ charge, population @ charge) / (charge @ population @ charge)
    susceptibility = constituent @ susceptibility_species @ constituent.T

    species_mobility = np.diag(diffusivity * np.array((1.0, 4.0, 2.0, NEUTRAL_MOBILITY)) * concentration)
    mobility_species = species_mobility - np.outer(species_mobility @ charge, species_mobility @ charge) / (charge @ species_mobility @ charge)
    mobility = constituent @ mobility_species @ constituent.T
    return susceptibility, mobility


def equilibrium_correlations(t, totals, harmonic, diffusivity):
    """Compatibility helper for callers using the previous module-level API."""
    susceptibility, mobility = fluctuation_matrices(totals, diffusivity)
    relaxation = (int(harmonic) * np.pi / CELL_LENGTH) ** 2 * np.linalg.solve(susceptibility, mobility.T).T
    return np.array([expm(-float(lag) * relaxation) @ susceptibility for lag in np.asarray(t)])
