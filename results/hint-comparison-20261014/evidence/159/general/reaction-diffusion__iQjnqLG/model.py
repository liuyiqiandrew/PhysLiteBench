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
    """Return free A, free B, free C, and bound P at local equilibrium."""
    values = np.asarray(total, dtype=float)
    if values.shape == () or values.shape[-1] != 2:
        raise ValueError("total must have a final dimension of length 2")

    a_total = np.maximum(values[..., 0], 0.0)
    b_total = np.maximum(values[..., 1], 0.0)
    upper = np.maximum(np.minimum(a_total, b_total), 0.0)
    bound = np.zeros_like(a_total)

    # The mass-action residual is monotone on [0, min(a_total,b_total)].
    # Safeguarding the Newton step keeps intermediate stiff-ODE states safe.
    for _ in range(40):
        free_a = a_total - bound
        free_b = b_total - bound
        free_c = free_a + free_b
        residual = bound - ASSOCIATION * free_a * free_b * free_c**2
        derivative = 1.0 + ASSOCIATION * (
            free_c**3 + 4.0 * free_a * free_b * free_c
        )
        next_bound = np.clip(bound - residual / derivative, 0.0, upper)
        if next_bound.size == 0 or np.max(np.abs(next_bound - bound)) < 1e-14:
            bound = next_bound
            break
        bound = next_bound

    free_a = a_total - bound
    free_b = b_total - bound
    free_c = free_a + free_b
    return np.stack((free_a, free_b, free_c, bound), axis=-1)


class TransportModel:
    def __init__(self):
        self.diffusivity = None

    @staticmethod
    def _binary_solution(t, x, initial, diffusivity):
        """Reflecting-boundary solution when only one cation is present."""
        t = np.asarray(t, dtype=float)
        x = np.asarray(x, dtype=float)
        initial = np.asarray(initial, dtype=float)
        n = len(x)
        if len(t) == 0:
            return np.empty((0, n, 2), dtype=float)
        if n == 1:
            return np.broadcast_to(initial, (len(t), 1, 2)).copy()

        eigenvalue = 4.0 / (x[1] - x[0]) ** 2 * np.sin(
            np.arange(n) * np.pi / (2.0 * (n - 1))
        ) ** 2
        coefficients = dct(initial, type=1, axis=0)
        rates = eigenvalue[:, None] * diffusivity * np.array([4.0 / 3.0, 8.0 / 3.0])
        evolved = np.exp(-t[:, None, None] * rates[None, :, :])
        return idct(evolved * coefficients[None, :, :], type=1, axis=1)

    def fit(self, data):
        """Fit the common diffusion scale from the calibration records."""
        x = np.asarray(data["x"], dtype=float)
        t = np.asarray(data["t"], dtype=float)
        initial = np.asarray(data["initial"], dtype=float)
        observations = np.asarray(data["concentration"], dtype=float)
        sigma = np.asarray(data["sigma"], dtype=float)

        if initial.ndim != 3 or initial.shape[-1] != 2:
            raise ValueError("initial must have shape (preparation, x, 2)")
        expected = (len(initial), len(t), len(x), 2)
        if observations.shape != expected or sigma.shape != expected:
            raise ValueError("calibration arrays have incompatible shapes")
        if not all(
            np.all(record[:, 0] == 0.0) or np.all(record[:, 1] == 0.0)
            for record in initial
        ):
            raise ValueError("calibration preparations must contain one cation")

        def objective(log_diffusivity):
            candidate = float(np.exp(log_diffusivity))
            value = 0.0
            for record, observed, error in zip(initial, observations, sigma):
                predicted = self._binary_solution(t, x, record, candidate)
                residual = (predicted - observed) / error
                value += float(np.sum(residual * residual))
            return value

        result = minimize_scalar(
            objective,
            bounds=(np.log(1e-13), np.log(1e-6)),
            method="bounded",
            options={"xatol": 1e-12, "maxiter": 100},
        )
        if not result.success or not np.isfinite(result.fun):
            raise RuntimeError("diffusivity fit failed")
        self.diffusivity = float(np.exp(result.x))
        return self

    def _binary(self, t, x, initial):
        if self.diffusivity is None:
            raise RuntimeError("fit the model before predicting")
        return self._binary_solution(t, x, initial, self.diffusivity)

    def predict(self, t, x, initial):
        """Predict the two total constituent concentration profiles."""
        t = np.asarray(t, dtype=float)
        x = np.asarray(x, dtype=float)
        initial = np.asarray(initial, dtype=float)
        if t.ndim != 1 or x.ndim != 1 or initial.shape != (len(x), 2):
            raise ValueError("expected t, x, and initial shapes (T,), (X,), (X,2)")
        if len(t) == 0:
            return np.empty((0, len(x), 2), dtype=float)
        if np.any(t < 0.0) or np.any(np.diff(t) < 0.0):
            raise ValueError("t must be increasing and nonnegative")
        if len(x) < 2 or np.any(np.diff(x) <= 0.0):
            raise ValueError("x must contain an increasing mesh with at least two points")
        if self.diffusivity is None:
            raise RuntimeError("fit the model before predicting")
        if t[-1] == 0.0:
            return np.broadcast_to(initial, (len(t), len(x), 2)).copy()

        # No cluster can form in a single-cation preparation; this exact path
        # is also the calibration model.
        if np.all(initial[:, 0] == 0.0) or np.all(initial[:, 1] == 0.0):
            return self._binary(t, x, initial)

        count = len(x)
        dx = x[1] - x[0]
        cell_width = np.full(count, dx)
        cell_width[[0, -1]] *= 0.5
        da, db, dc = self.diffusivity * np.array([1.0, 4.0, 2.0])

        def derivative(_time, flat_state):
            state = flat_state.reshape(count, 2)
            species = equilibrium(state)
            face_species = 0.5 * (species[:-1] + species[1:])
            gradient = np.diff(species, axis=0) / dx

            # Choose the field so the instantaneous electric current is zero.
            numerator = da * gradient[:, 0] + db * gradient[:, 1] - dc * gradient[:, 2]
            denominator = (
                da * face_species[:, 0]
                + db * face_species[:, 1]
                + dc * face_species[:, 2]
            )
            potential_gradient = np.divide(
                -numerator,
                denominator,
                out=np.zeros_like(numerator),
                where=denominator > 0.0,
            )
            free_flux = -np.array([da, db]) * (
                gradient[:, :2] + face_species[:, :2] * potential_gradient[:, None]
            )
            bound_flux = -self.diffusivity * NEUTRAL_MOBILITY * gradient[:, 3]

            flux = np.zeros((count + 1, 2), dtype=float)
            flux[1:-1] = free_flux + bound_flux[:, None]
            return (-np.diff(flux, axis=0) / cell_width[:, None]).ravel()

        state_size = 2 * count
        offsets = np.arange(-3, 4)
        sparsity = diags(
            [np.ones(state_size - abs(offset)) for offset in offsets],
            offsets,
            shape=(state_size, state_size),
            format="csr",
        )
        result = solve_ivp(
            derivative,
            (0.0, float(t[-1])),
            initial.ravel(),
            method="BDF",
            t_eval=t,
            rtol=1e-8,
            atol=1e-10,
            jac_sparsity=sparsity,
        )
        if not result.success:
            raise RuntimeError(result.message)
        prediction = result.y.T.reshape(len(t), count, 2)
        prediction[t == 0.0] = initial
        return prediction


def fluctuation_matrices(totals, diffusivity):
    """Return the charge-constrained susceptibility and mobility matrices."""
    totals = np.asarray(totals, dtype=float)
    if totals.shape != (2,) or np.any(totals <= 0.0):
        raise ValueError("totals must contain two positive concentrations")
    concentration = equilibrium(totals)

    charge = np.array([1.0, 1.0, -1.0, 0.0])
    constituents = np.array([[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0]])
    population = np.diag(concentration)
    charge_population = population @ charge
    conditional = population - np.outer(charge_population, charge_population) / (
        charge @ population @ charge
    )
    susceptibility = constituents @ conditional @ constituents.T

    species_mobility = np.diag(
        diffusivity * np.array([1.0, 4.0, 2.0, NEUTRAL_MOBILITY]) * concentration
    )
    mobile_charge = species_mobility @ charge
    constrained = species_mobility - np.outer(mobile_charge, mobile_charge) / (
        charge @ species_mobility @ charge
    )
    mobility = constituents @ constrained @ constituents.T
    return susceptibility, mobility


def equilibrium_correlations(t, totals, harmonic, diffusivity):
    """Compute the normalized equilibrium mode covariance."""
    t = np.asarray(t, dtype=float)
    susceptibility, mobility = fluctuation_matrices(totals, diffusivity)
    relaxation = (
        (harmonic * np.pi / CELL_LENGTH) ** 2
        * mobility
        @ np.linalg.inv(susceptibility)
    )
    # This is the prescribed equal-time covariance of total constituents.
    initial_covariance = np.diag(np.asarray(totals, dtype=float))
    if len(t) == 0:
        return np.empty((0, 2, 2), dtype=float)
    return np.stack(
        [expm(-float(time) * relaxation) @ initial_covariance for time in t],
        axis=0,
    )


def _correlate(self, t, totals, harmonic=1):
    if self.diffusivity is None:
        raise RuntimeError("fit the model before computing correlations")
    times = np.asarray(t, dtype=float)
    if times.ndim != 1 or np.any(times < 0.0) or np.any(np.diff(times) < 0.0):
        raise ValueError("t must be an increasing, nonnegative one-dimensional array")
    if not isinstance(harmonic, (int, np.integer)) or not 1 <= harmonic <= 8:
        raise ValueError("harmonic must be an integer from 1 through 8")
    return equilibrium_correlations(times, np.asarray(totals, dtype=float), harmonic, self.diffusivity)


TransportModel.correlate = _correlate
