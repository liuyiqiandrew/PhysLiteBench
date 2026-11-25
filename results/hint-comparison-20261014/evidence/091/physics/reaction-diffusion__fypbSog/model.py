"""Electroneutral, locally equilibrated transport for the electrolyte cell.

The state variables used here are the two *total* cation constituents. Free
ion and cluster concentrations are reconstructed at every spatial point from
the mass-action equilibrium. This avoids introducing an arbitrary fast
reaction rate into either the fit or the prediction.
"""

import numpy as np
from scipy.fft import dct, idct
from scipy.integrate import solve_ivp
from scipy.optimize import minimize_scalar


ASSOCIATION = 0.5
NEUTRAL_MOBILITY = 0.15
CELL_LENGTH = 1.0e-3


def equilibrium(totals):
    """Return ``(a, b, c, p)`` at mass-action equilibrium."""
    totals = np.asarray(totals, dtype=float)
    if totals.shape[-1] != 2:
        raise ValueError("totals must have a final A,B dimension of length 2")
    a_total = totals[..., 0]
    b_total = totals[..., 1]
    # Valid states are non-negative. The clipping also tolerates a very small
    # negative trial state from the implicit integrator.
    a_work = np.maximum(a_total, 0.0)
    b_work = np.maximum(b_total, 0.0)
    lo = np.zeros_like(a_work)
    hi = np.minimum(a_work, b_work)
    for _ in range(70):
        p = 0.5 * (lo + hi)
        a = a_work - p
        b = b_work - p
        c = a + b
        # p - K*a*b*c^2 is monotonic increasing.
        below_root = p < ASSOCIATION * a * b * c * c
        lo = np.where(below_root, p, lo)
        hi = np.where(below_root, hi, p)
    p = 0.5 * (lo + hi)
    return np.stack((a_work - p, b_work - p, a_work + b_work - 2.0 * p, p), axis=-1)


def _validate_grid(t, x, initial):
    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)
    initial = np.asarray(initial, dtype=float)
    if t.ndim != 1 or len(t) == 0 or np.any(~np.isfinite(t)) or t[0] < 0 or np.any(np.diff(t) < 0):
        raise ValueError("t must be a nonempty increasing array of nonnegative times")
    if x.ndim != 1 or len(x) < 2 or np.any(~np.isfinite(x)):
        raise ValueError("x must contain at least two finite points")
    dx = np.diff(x)
    if np.any(dx <= 0) or not np.allclose(dx, dx[0], rtol=1e-10, atol=1e-14):
        raise ValueError("x must be an equally spaced increasing mesh")
    if initial.shape != (len(x), 2) or not np.all(np.isfinite(initial)):
        raise ValueError("initial must have shape (len(x), 2) and be finite")
    return t, x, initial


class TransportModel:
    def __init__(self):
        self.diffusivity = None

    @staticmethod
    def _binary_prediction(t, x, initial, diffusivity):
        """Exact semi-discrete Neumann solution for a one-cation preparation."""
        n_intervals = len(x) - 1
        dx = x[1] - x[0]
        laplace_eigenvalues = 4.0 / dx**2 * np.sin(
            np.arange(n_intervals + 1) * np.pi / (2.0 * n_intervals)
        ) ** 2
        modes = dct(initial, type=1, axis=0)
        # Zero-current ambipolar diffusivities for A+/C- and B+/C-.
        ambipolar = diffusivity * np.array((4.0 / 3.0, 8.0 / 3.0))
        attenuation = np.exp(-t[:, None, None] * laplace_eigenvalues[None, :, None] * ambipolar)
        return idct(attenuation * modes[None, :, :], type=1, axis=1)

    def fit(self, data):
        """Fit the sole dimensional parameter, the A+ diffusion coefficient."""
        try:
            t = np.asarray(data["t"], dtype=float)
            x = np.asarray(data["x"], dtype=float)
            initial = np.asarray(data["initial"], dtype=float)
            observed = np.asarray(data["concentration"], dtype=float)
            sigma = np.asarray(data["sigma"], dtype=float)
        except (KeyError, TypeError) as error:
            raise ValueError("data must provide x, t, initial, concentration, and sigma") from error
        if initial.ndim != 3 or initial.shape[1:] != (len(x), 2):
            raise ValueError("initial must have shape (records, len(x), 2)")
        if observed.shape != (len(initial), len(t), len(x), 2) or sigma.shape != observed.shape:
            raise ValueError("concentration and sigma have incompatible shapes")
        if np.any(sigma <= 0) or not np.all(np.isfinite(sigma)):
            raise ValueError("sigma must be finite and positive")
        _validate_grid(t, x, initial[0])
        if not (np.all(np.isfinite(observed)) and np.all(np.isfinite(initial))):
            raise ValueError("calibration arrays must be finite")
        # The documented calibration records are binary. Their analytic
        # ambipolar solution keeps discretization of the fast equilibrium out
        # of the regression.
        if not np.all((np.max(np.abs(initial[:, :, 0]), axis=1) == 0.0) |
                      (np.max(np.abs(initial[:, :, 1]), axis=1) == 0.0)):
            raise ValueError("calibration preparations must contain only one cation constituent")

        def chi_squared(log_diffusivity):
            diffusivity = float(np.exp(log_diffusivity))
            prediction = np.stack([
                self._binary_prediction(t, x, preparation, diffusivity) for preparation in initial
            ])
            residual = (prediction - observed) / sigma
            return float(np.sum(residual * residual))

        result = minimize_scalar(
            chi_squared, bounds=(np.log(1e-14), np.log(1e-5)), method="bounded",
            options={"xatol": 1e-11},
        )
        if not result.success or not np.isfinite(result.fun):
            raise RuntimeError("could not fit the diffusion coefficient")
        self.diffusivity = float(np.exp(result.x))
        return self

    def predict(self, t, x, initial):
        """Predict total A and B profiles at the requested times and nodes."""
        if self.diffusivity is None or not np.isfinite(self.diffusivity) or self.diffusivity <= 0:
            raise RuntimeError("fit must be called before predict")
        t, x, initial = _validate_grid(t, x, initial)
        if np.all(initial[:, 0] == 0.0) or np.all(initial[:, 1] == 0.0):
            return self._binary_prediction(t, x, initial, self.diffusivity)
        if t[-1] == 0.0:
            return np.broadcast_to(initial, (len(t),) + initial.shape).copy()

        count = len(x)
        dx = x[1] - x[0]
        volumes = np.full(count, dx)
        volumes[[0, -1]] *= 0.5
        d_a, d_b, d_c = self.diffusivity * np.array((1.0, 4.0, 2.0))

        def rhs(_time, flat_total):
            total = flat_total.reshape(count, 2)
            species = equilibrium(total)
            face_species = 0.5 * (species[:-1] + species[1:])
            gradient = np.diff(species, axis=0) / dx
            # This field makes the charge current through every face zero.
            field_numerator = d_a * gradient[:, 0] + d_b * gradient[:, 1] - d_c * gradient[:, 2]
            field_denominator = d_a * face_species[:, 0] + d_b * face_species[:, 1] + d_c * face_species[:, 2]
            # A completely empty face has no charge current to cancel.
            field_gradient = -np.divide(
                field_numerator, field_denominator, out=np.zeros_like(field_numerator),
                where=field_denominator > 0.0,
            )
            ion_flux = -np.column_stack((d_a, d_b)) * (
                gradient[:, :2] + face_species[:, :2] * field_gradient[:, None]
            )
            cluster_flux = -NEUTRAL_MOBILITY * self.diffusivity * gradient[:, 3]
            constituent_face_flux = ion_flux + cluster_flux[:, None]
            flux = np.zeros((count + 1, 2))  # blocking end faces
            flux[1:-1] = constituent_face_flux
            return (-np.diff(flux, axis=0) / volumes[:, None]).ravel()

        solution = solve_ivp(
            rhs, (0.0, float(t[-1])), initial.ravel(), t_eval=t, method="BDF",
            rtol=2e-8, atol=1e-10,
        )
        if not solution.success:
            raise RuntimeError(solution.message)
        return solution.y.T.reshape(len(t), count, 2)

    def correlate(self, t, totals, harmonic=1):
        """Return the normalized equilibrium covariance for one cosine mode."""
        if self.diffusivity is None or not np.isfinite(self.diffusivity) or self.diffusivity <= 0:
            raise RuntimeError("fit must be called before correlate")
        t = np.asarray(t, dtype=float)
        totals = np.asarray(totals, dtype=float)
        if t.ndim != 1 or len(t) == 0 or np.any(~np.isfinite(t)) or t[0] < 0 or np.any(np.diff(t) < 0):
            raise ValueError("t must be a nonempty increasing array of nonnegative times")
        if totals.shape != (2,) or np.any(~np.isfinite(totals)) or np.any(totals <= 0):
            raise ValueError("totals must be two positive finite concentrations")
        if isinstance(harmonic, (bool, np.bool_)) or int(harmonic) != harmonic or not 1 <= int(harmonic) <= 8:
            raise ValueError("harmonic must be an integer from 1 through 8")
        harmonic = int(harmonic)
        susceptibility, mobility = fluctuation_matrices(totals, self.diffusivity)
        # In the C^(-1/2) basis the rate is symmetric. This preserves exact
        # zero-lag susceptibility and Onsager reciprocity numerically.
        values, vectors = np.linalg.eigh(susceptibility)
        root = (vectors * np.sqrt(values)) @ vectors.T
        inverse_root = (vectors * (1.0 / np.sqrt(values))) @ vectors.T
        symmetric_rate = inverse_root @ mobility @ inverse_root
        rates, rate_vectors = np.linalg.eigh(symmetric_rate)
        wave_number_squared = (harmonic * np.pi / CELL_LENGTH) ** 2
        factors = np.exp(-t[:, None] * wave_number_squared * rates[None, :])
        return np.array([root @ ((rate_vectors * factor) @ rate_vectors.T) @ root for factor in factors])


def fluctuation_matrices(totals, diffusivity):
    """Constrained equilibrium susceptibility and total-constituent mobility."""
    species = equilibrium(np.asarray(totals, dtype=float))
    if species.shape != (4,):
        raise ValueError("totals must contain exactly A and B")
    charge = np.array((1.0, 1.0, -1.0, 0.0))
    constituents = np.array(((1.0, 0.0, 0.0, 1.0), (0.0, 1.0, 0.0, 1.0)))
    ideal_covariance = np.diag(species)
    constrained_covariance = ideal_covariance - np.outer(
        ideal_covariance @ charge, charge @ ideal_covariance
    ) / (charge @ ideal_covariance @ charge)
    susceptibility = constituents @ constrained_covariance @ constituents.T

    particle_mobility = np.diag(
        diffusivity * np.array((1.0, 4.0, 2.0, NEUTRAL_MOBILITY)) * species
    )
    constrained_mobility = particle_mobility - np.outer(
        particle_mobility @ charge, charge @ particle_mobility
    ) / (charge @ particle_mobility @ charge)
    mobility = constituents @ constrained_mobility @ constituents.T
    return susceptibility, mobility


def equilibrium_correlations(t, totals, harmonic, diffusivity):
    """Compatibility helper implementing the same calculation as correlate."""
    model = TransportModel()
    model.diffusivity = float(diffusivity)
    return model.correlate(t, totals, harmonic)
