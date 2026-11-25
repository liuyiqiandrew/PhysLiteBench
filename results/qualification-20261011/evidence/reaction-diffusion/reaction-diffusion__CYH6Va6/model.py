"""Transport and equilibrium-fluctuation model for the electrolyte cell."""

import numpy as np
from scipy.fft import dct, idct
from scipy.integrate import solve_ivp
from scipy.linalg import expm
from scipy.optimize import minimize_scalar
from scipy.sparse import diags


# Neutral four-ion association A+ + B+ + 2 C- <-> P.
ASSOCIATION = 0.5
NEUTRAL_MOBILITY = 0.15
CELL_LENGTH = 0.001


def equilibrium(total):
    """Return ``(c_A, c_B, c_C, p)`` at local association equilibrium."""
    values = np.asarray(total, dtype=float)
    if values.ndim == 0 or values.shape[-1] != 2:
        raise ValueError("total must have a final dimension of length 2")

    shape = values.shape[:-1]
    flat = np.maximum(values.reshape(-1, 2), 0.0)
    a, b = flat.T
    upper = np.minimum(a, b)
    bound = np.zeros_like(a)

    # f(p) is monotone increasing on [0, min(a,b)].  Projected Newton is
    # quick for the many local equilibrium evaluations in the ODE solver.
    for _ in range(40):
        free_a = a - bound
        free_b = b - bound
        anion = free_a + free_b
        value = bound - ASSOCIATION * free_a * free_b * anion**2
        derivative = 1.0 + ASSOCIATION * (
            anion**3 + 4.0 * free_a * free_b * anion
        )
        change = value / derivative
        next_bound = np.clip(bound - change, 0.0, upper)
        if np.max(np.abs(next_bound - bound), initial=0.0) < 1e-14:
            bound = next_bound
            break
        bound = next_bound

    result = np.column_stack((a - bound, b - bound, a + b - 2.0 * bound, bound))
    return result.reshape(shape + (4,))


def _neumann_rates(x):
    """Eigenvalue factors for the endpoint-inclusive Neumann mesh."""
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or x.size < 2:
        raise ValueError("x must contain at least two mesh points")
    dx = np.diff(x)
    if not np.all(np.isfinite(x)) or np.any(dx <= 0.0):
        raise ValueError("x must be a strictly increasing mesh")
    if not np.allclose(dx, dx[0], rtol=1e-9, atol=1e-15):
        raise ValueError("x must be equally spaced")
    count = x.size - 1
    modes = np.arange(count + 1, dtype=float)
    return 4.0 / dx[0] ** 2 * np.sin(modes * np.pi / (2.0 * count)) ** 2


class TransportModel:
    def __init__(self):
        self.diffusivity = None

    def fit(self, data):
        """Fit the common diffusion scale from the single-cation records."""
        x = np.asarray(data["x"], dtype=float)
        t = np.asarray(data["t"], dtype=float).reshape(-1)
        initial = np.asarray(data["initial"], dtype=float)
        observed = np.asarray(data["concentration"], dtype=float)
        sigma = np.asarray(data.get("sigma", np.ones_like(observed)), dtype=float)

        if initial.ndim != 3 or initial.shape[-1] != 2:
            raise ValueError("initial must have shape (preparation, x, 2)")
        if observed.shape[:2] != (initial.shape[0], t.size):
            raise ValueError("concentration has incompatible preparation/time axes")
        if observed.shape[-2:] != initial.shape[1:]:
            raise ValueError("concentration has incompatible spatial/channel axes")
        if sigma.shape != observed.shape or np.any(sigma <= 0.0):
            raise ValueError("sigma must be positive and match concentration")

        # With one cation absent there is no association and the active
        # constituent obeys scalar ambipolar diffusion.
        pure = np.logical_or(
            np.all(np.isclose(initial[:, :, 0], 0.0, atol=1e-14), axis=1),
            np.all(np.isclose(initial[:, :, 1], 0.0, atol=1e-14), axis=1),
        )
        if not np.any(pure):
            raise ValueError("calibration data contain no single-cation preparation")
        calibration = [
            (initial[i], observed[i], sigma[i]) for i in np.flatnonzero(pure)
        ]

        # Optimize in log space so positivity is automatic.
        log_bounds = (np.log(1e-14), np.log(1e-5))

        def objective(log_diffusivity):
            self.diffusivity = float(np.exp(log_diffusivity))
            total = 0.0
            for preparation, measured, uncertainty in calibration:
                predicted = self._binary(t, x, preparation)
                residual = (predicted - measured) / uncertainty
                total += float(np.sum(residual * residual))
            return total

        result = minimize_scalar(
            objective,
            bounds=log_bounds,
            method="bounded",
            options={"xatol": 1e-10, "maxiter": 200},
        )
        if not result.success or not np.isfinite(result.x):
            raise RuntimeError("unable to fit the diffusion scale")
        self.diffusivity = float(np.exp(result.x))
        if not np.isfinite(self.diffusivity) or self.diffusivity <= 0.0:
            raise RuntimeError("fitted diffusion scale is not positive")
        return self

    def _binary(self, t, x, initial):
        """Exact Neumann solution for preparations with one absent cation."""
        t = np.asarray(t, dtype=float).reshape(-1)
        x = np.asarray(x, dtype=float)
        initial = np.asarray(initial, dtype=float)
        if self.diffusivity is None:
            raise RuntimeError("fit must be called before prediction")
        if initial.shape != (x.size, 2):
            raise ValueError("initial must have shape (len(x), 2)")
        if t.size == 0:
            return np.empty((0, x.size, 2), dtype=float)

        spatial_rate = _neumann_rates(x)
        coefficients = dct(initial, type=1, axis=0)
        ambipolar = self.diffusivity * np.array([4.0 / 3.0, 8.0 / 3.0])
        rates = spatial_rate[:, None] * ambipolar[None, :]
        modes = np.exp(-t[:, None, None] * rates[None, :, :])
        return idct(modes * coefficients[None, :, :], type=1, axis=1)

    def predict(self, t, x, initial):
        """Predict the two total constituent concentration profiles."""
        t = np.asarray(t, dtype=float).reshape(-1)
        x = np.asarray(x, dtype=float)
        initial = np.asarray(initial, dtype=float)
        if self.diffusivity is None:
            raise RuntimeError("fit must be called before prediction")
        if initial.shape != (x.size, 2):
            raise ValueError("initial must have shape (len(x), 2)")
        if t.size == 0:
            return np.empty((0, x.size, 2), dtype=float)
        if np.any(~np.isfinite(t)) or np.any(t < 0.0):
            raise ValueError("t must contain finite nonnegative times")
        if np.any(np.diff(t) < 0.0):
            raise ValueError("t must be increasing")

        if np.allclose(initial[:, 0], 0.0, atol=1e-14) or np.allclose(
            initial[:, 1], 0.0, atol=1e-14
        ):
            return self._binary(t, x, initial)
        if t[-1] == 0.0:
            return initial[None, :, :].copy()

        _neumann_rates(x)  # validate the mesh before integrating
        count = x.size
        dx = x[1] - x[0]
        widths = np.full(count, dx)
        widths[[0, -1]] *= 0.5
        da, db, dc = self.diffusivity * np.array([1.0, 4.0, 2.0])

        def derivative(_time, flat_state):
            totals = flat_state.reshape(count, 2)
            species = equilibrium(totals)
            face_species = 0.5 * (species[:-1] + species[1:])
            gradient = np.diff(species, axis=0) / dx

            # Select the electric field that cancels the instantaneous free
            # ion charge current at each face.
            numerator = (
                da * gradient[:, 0]
                + db * gradient[:, 1]
                - dc * gradient[:, 2]
            )
            denominator = (
                da * face_species[:, 0]
                + db * face_species[:, 1]
                + dc * face_species[:, 2]
            )
            field_gradient = np.divide(
                -numerator,
                denominator,
                out=np.zeros_like(numerator),
                where=denominator > 0.0,
            )
            free_flux = -np.array([da, db]) * (
                gradient[:, :2] + face_species[:, :2] * field_gradient[:, None]
            )
            bound_flux = -self.diffusivity * NEUTRAL_MOBILITY * gradient[:, 3]

            flux = np.zeros((count + 1, 2), dtype=float)
            flux[1:-1] = free_flux + bound_flux[:, None]
            return (-np.diff(flux, axis=0) / widths[:, None]).ravel()

        dimension = 2 * count
        offsets = np.arange(-3, 4)
        sparsity = diags(
            [np.ones(dimension - abs(offset)) for offset in offsets],
            offsets,
            shape=(dimension, dimension),
            format="csr",
        )
        result = solve_ivp(
            derivative,
            (0.0, float(t[-1])),
            initial.ravel(),
            method="BDF",
            t_eval=t,
            rtol=2e-8,
            atol=1e-10,
            jac_sparsity=sparsity,
        )
        if not result.success:
            raise RuntimeError(result.message)
        return result.y.T.reshape(t.size, count, 2)

    def correlate(self, t, totals, harmonic=1):
        """Return the equilibrium two-time covariance of a spatial mode."""
        return equilibrium_correlations(t, totals, harmonic, self.diffusivity)


def fluctuation_matrices(totals, diffusivity):
    """Build the equilibrium susceptibility and constrained mobility."""
    totals = np.asarray(totals, dtype=float)
    if totals.shape != (2,) or np.any(~np.isfinite(totals)) or np.any(totals <= 0.0):
        raise ValueError("totals must be two positive finite concentrations")
    if not np.isfinite(diffusivity) or diffusivity <= 0.0:
        raise ValueError("diffusivity must be positive")

    concentration = equilibrium(totals)
    charge = np.array([1.0, 1.0, -1.0, 0.0])
    constituents = np.array(
        [[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0]]
    )
    population = np.diag(concentration)
    charge_variance = charge @ population @ charge
    conditional = population - np.outer(
        population @ charge, population @ charge
    ) / charge_variance
    susceptibility = constituents @ conditional @ constituents.T

    species_mobility = np.diag(
        diffusivity * np.array([1.0, 4.0, 2.0, NEUTRAL_MOBILITY]) * concentration
    )
    mobility_charge = species_mobility @ charge
    current_variance = charge @ species_mobility @ charge
    constrained = species_mobility - np.outer(
        mobility_charge, mobility_charge
    ) / current_variance
    mobility = constituents @ constrained @ constituents.T
    return susceptibility, mobility


def equilibrium_correlations(t, totals, harmonic, diffusivity):
    """Evaluate the prescribed mode covariance at each nonnegative lag."""
    times = np.asarray(t, dtype=float).reshape(-1)
    if np.any(~np.isfinite(times)) or np.any(times < 0.0):
        raise ValueError("t must contain finite nonnegative times")
    if np.any(np.diff(times) < 0.0):
        raise ValueError("t must be increasing")
    if int(harmonic) != harmonic or not 1 <= int(harmonic) <= 8:
        raise ValueError("harmonic must be an integer from 1 through 8")

    susceptibility, mobility = fluctuation_matrices(totals, diffusivity)
    relaxation = (
        (int(harmonic) * np.pi / CELL_LENGTH) ** 2
        * mobility
        @ np.linalg.inv(susceptibility)
    )
    # This is the normalization specified for the supplied predictor.
    initial_covariance = np.diag(np.asarray(totals, dtype=float))
    if times.size == 0:
        return np.empty((0, 2, 2), dtype=float)
    return np.array(
        [expm(-float(time) * relaxation) @ initial_covariance for time in times]
    )
