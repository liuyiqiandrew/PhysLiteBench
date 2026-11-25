"""Transport model for the electroneutral, fast-association cell.

The state used by :class:`TransportModel` is the pair of *total*
constituent concentrations.  At every evaluation the free concentrations and
the concentration of P are recovered from the local mass-action equilibrium.
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


def equilibrium(totals):
    """Return free A, free B, free C, and P for total A/B concentrations."""
    q = np.asarray(totals, dtype=float)
    if q.shape[-1] != 2:
        raise ValueError("totals must have a final dimension of length two")
    a_total = np.maximum(q[..., 0], 0.0)
    b_total = np.maximum(q[..., 1], 0.0)
    upper = np.minimum(a_total, b_total)
    p = np.zeros_like(a_total)

    # f(p) = p - K (a-p)(b-p)(a+b-2p)^2.
    for _ in range(40):
        a = a_total - p
        b = b_total - p
        c = a + b
        value = p - ASSOCIATION * a * b * c**2
        slope = 1.0 + ASSOCIATION * (c**3 + 4.0 * a * b * c)
        next_p = np.clip(p - value / slope, 0.0, upper)
        if np.max(np.abs(next_p - p), initial=0.0) < 2e-14:
            p = next_p
            break
        p = next_p

    a = a_total - p
    b = b_total - p
    return np.stack((a, b, a + b, p), axis=-1)


class TransportModel:
    """Fit and evaluate the one-parameter transport model."""

    def __init__(self):
        self.diffusivity = None

    @staticmethod
    def _validate_mesh(x):
        x = np.asarray(x, dtype=float)
        if x.ndim != 1 or x.size < 2:
            raise ValueError("x must be a one-dimensional mesh with at least two points")
        dx = x[1] - x[0]
        if not np.isfinite(dx) or dx <= 0 or not np.allclose(np.diff(x), dx):
            raise ValueError("x must be an equally spaced increasing mesh")
        return x, dx

    def fit(self, data):
        """Fit D by weighted least squares over the supplied calibrations."""
        x, _ = self._validate_mesh(data["x"])
        time = np.asarray(data["t"], dtype=float)
        initials = np.asarray(data["initial"], dtype=float)
        observed = np.asarray(data["concentration"], dtype=float)
        sigma = np.asarray(data["sigma"], dtype=float)
        if (initials.ndim != 3 or initials.shape[-1] != 2 or
                observed.shape != (initials.shape[0], time.size, x.size, 2) or
                sigma.shape != observed.shape or np.any(sigma <= 0)):
            raise ValueError("calibration data have incompatible shapes or uncertainties")

        # Optimise log(D), both to enforce positivity and to resolve small SI
        # diffusivities accurately.
        def objective(log_d):
            self.diffusivity = float(np.exp(log_d))
            residual_sum = 0.0
            for initial, measurement, error in zip(initials, observed, sigma):
                predicted = self.predict(time, x, initial)
                residual = (predicted - measurement) / error
                residual_sum += float(np.sum(residual * residual))
            return residual_sum if np.isfinite(residual_sum) else np.inf

        result = minimize_scalar(
            objective,
            bounds=(np.log(1e-13), np.log(1e-6)), method="bounded",
            options={"xatol": 1e-11},
        )
        if not result.success or not np.isfinite(result.fun):
            raise RuntimeError("could not fit diffusivity")
        self.diffusivity = float(np.exp(result.x))
        return self

    def _binary(self, time, x, initial):
        """Exact discrete Neumann solution for an A-only or B-only record."""
        n_interval = len(x) - 1
        eigenvalues = (4.0 / (x[1] - x[0]) ** 2
                       * np.sin(np.arange(n_interval + 1) * np.pi / (2.0 * n_interval)) ** 2)
        modes = dct(initial, type=1, axis=0)
        # Zero-current ambipolar diffusivities for A/C and B/C respectively.
        rates = eigenvalues[:, None] * self.diffusivity * np.array([4.0 / 3.0, 8.0 / 3.0])
        return idct(np.exp(-time[:, None, None] * rates[None, :, :]) * modes[None, :, :],
                    type=1, axis=1)

    def predict(self, t, x, initial):
        """Predict total A and B profiles at each requested age."""
        if self.diffusivity is None or not np.isfinite(self.diffusivity) or self.diffusivity <= 0:
            raise RuntimeError("fit must be called before predict")
        time = np.asarray(t, dtype=float)
        x, dx = self._validate_mesh(x)
        initial = np.asarray(initial, dtype=float)
        if (time.ndim != 1 or time.size == 0 or np.any(time < 0) or
                np.any(np.diff(time) < 0) or initial.shape != (x.size, 2)):
            raise ValueError("invalid time array or initial concentration shape")
        if not np.isfinite(initial).all() or np.any(initial < 0):
            raise ValueError("initial concentrations must be finite and nonnegative")

        # solve_ivp does not construct an output array for a zero-length time
        # span, whereas the preparation itself is a valid requested result.
        if time[-1] == 0.0:
            return np.repeat(initial[None, :, :], time.size, axis=0)

        # A calibration preparation has no P at any time.  This exact route
        # also avoids numerical integration in the fitting inner loop.
        if np.all(initial[:, 0] == 0.0) or np.all(initial[:, 1] == 0.0):
            return self._binary(time, x, initial)

        n = x.size
        volume_weights = np.full(n, dx)
        volume_weights[[0, -1]] *= 0.5
        d_a, d_b, d_c = self.diffusivity * np.array([1.0, 4.0, 2.0])

        def rhs(_age, flat):
            species = equilibrium(flat.reshape(n, 2))
            face = 0.5 * (species[:-1] + species[1:])
            gradient = np.diff(species, axis=0) / dx

            # Charge-current-free electric-field gradient.  Empty faces have
            # zero numerator too, so choosing a zero field there is harmless.
            numerator = d_a * gradient[:, 0] + d_b * gradient[:, 1] - d_c * gradient[:, 2]
            denominator = d_a * face[:, 0] + d_b * face[:, 1] + d_c * face[:, 2]
            field = np.divide(-numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0)

            ion_flux = -np.array([d_a, d_b]) * (gradient[:, :2] + face[:, :2] * field[:, None])
            cluster_flux = -self.diffusivity * NEUTRAL_MOBILITY * gradient[:, 3]
            total_flux = np.zeros((n + 1, 2))  # no-flux boundaries
            total_flux[1:-1] = ion_flux + cluster_flux[:, None]
            return (-np.diff(total_flux, axis=0) / volume_weights[:, None]).ravel()

        # Each node couples only to itself and its two neighbours (two state
        # variables each), so this band is a useful BDF sparsity hint.
        sparsity = diags([np.ones(2 * n - abs(k)) for k in range(-3, 4)], range(-3, 4), format="csr")
        solution = solve_ivp(
            rhs, (0.0, float(time[-1])), initial.ravel(), method="BDF", t_eval=time,
            rtol=2e-8, atol=2e-10, jac_sparsity=sparsity,
        )
        if not solution.success:
            raise RuntimeError(solution.message)
        output = solution.y.T.reshape(time.size, n, 2)
        if not np.isfinite(output).all():
            raise RuntimeError("non-finite transport prediction")
        return output

    def correlate(self, t, totals, harmonic=1):
        """Return the equilibrium two-time covariance for one cosine mode."""
        if self.diffusivity is None or not np.isfinite(self.diffusivity) or self.diffusivity <= 0:
            raise RuntimeError("fit must be called before correlate")
        time = np.asarray(t, dtype=float)
        totals = np.asarray(totals, dtype=float)
        if time.ndim != 1 or time.size == 0 or np.any(time < 0) or np.any(np.diff(time) < 0):
            raise ValueError("t must be an increasing nonnegative array")
        if totals.shape != (2,) or not np.isfinite(totals).all() or np.any(totals <= 0):
            raise ValueError("totals must contain two positive concentrations")
        if isinstance(harmonic, bool) or int(harmonic) != harmonic or not 1 <= int(harmonic) <= 8:
            raise ValueError("harmonic must be an integer from 1 through 8")

        susceptibility, mobility = fluctuation_matrices(totals, self.diffusivity)
        q_squared = (int(harmonic) * np.pi / 0.001) ** 2
        relaxation = q_squared * np.linalg.solve(susceptibility.T, mobility.T).T

        # The specified detector convention assigns independent ideal total
        # constituent number fluctuations at equal time.
        equal_time = np.diag(totals)
        return np.array([expm(-float(age) * relaxation) @ equal_time for age in time])


def fluctuation_matrices(totals, diffusivity):
    """Thermodynamic susceptibility and zero-current Onsager mobility."""
    concentration = equilibrium(np.asarray(totals, dtype=float)[None, :])[0]
    charge = np.array([1.0, 1.0, -1.0, 0.0])
    constituents = np.array([[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0]])
    population = np.diag(concentration)
    conditional = population - np.outer(population @ charge, population @ charge) / (charge @ population @ charge)
    susceptibility = constituents @ conditional @ constituents.T

    species_mobility = np.diag(diffusivity * np.array([1.0, 4.0, 2.0, NEUTRAL_MOBILITY]) * concentration)
    mobile_charge = species_mobility @ charge
    constrained = species_mobility - np.outer(mobile_charge, mobile_charge) / (charge @ species_mobility @ charge)
    mobility = constituents @ constrained @ constituents.T
    return susceptibility, mobility


# Retained as a module-level helper for callers using the original scaffold.
def equilibrium_correlations(t, totals, harmonic, diffusivity):
    susceptibility, mobility = fluctuation_matrices(totals, diffusivity)
    relaxation = (harmonic * np.pi / 0.001) ** 2 * np.linalg.solve(susceptibility.T, mobility.T).T
    return np.array([expm(-float(age) * relaxation) @ np.diag(totals) for age in t])
