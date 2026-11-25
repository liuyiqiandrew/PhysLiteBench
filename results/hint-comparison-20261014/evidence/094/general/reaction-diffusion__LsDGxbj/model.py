"""Transport and equilibrium-fluctuation model for the electrolyte cell."""

import numpy as np
from scipy.fft import dct, idct
from scipy.integrate import solve_ivp
from scipy.linalg import expm
from scipy.optimize import minimize_scalar
from scipy.sparse import diags


# A + B + 2 C <-> P
ASSOCIATION = 0.5
NEUTRAL_MOBILITY = 0.15
CELL_LENGTH = 0.001


def equilibrium(total):
    """Return free A, free B, free C, and P at local equilibrium."""
    total = np.asarray(total, dtype=float)
    if total.ndim == 0 or total.shape[-1] != 2:
        raise ValueError("total must have a final dimension of length 2")

    a_total = total[..., 0]
    b_total = total[..., 1]
    bound = np.zeros_like(a_total)
    upper = np.minimum(a_total, b_total)

    # Newton's method is particularly well behaved here: the mass-action
    # residual is strictly increasing on [0, min(a_total,b_total)].
    for _ in range(64):
        free_a = a_total - bound
        free_b = b_total - bound
        free_c = free_a + free_b
        residual = bound - ASSOCIATION * free_a * free_b * free_c**2
        derivative = 1.0 + ASSOCIATION * (
            free_c**3 + 4.0 * free_a * free_b * free_c
        )
        step = residual / derivative
        updated = np.clip(bound - step, 0.0, upper)
        if np.all(np.abs(updated - bound) <= 2e-14 * (1.0 + np.abs(updated))):
            bound = updated
            break
        bound = updated

    free_a = a_total - bound
    free_b = b_total - bound
    return np.stack((free_a, free_b, free_a + free_b, bound), axis=-1)


def _check_mesh(x):
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or len(x) < 2 or not np.isfinite(x).all():
        raise ValueError("x must be a finite one-dimensional mesh with at least two points")
    spacing = np.diff(x)
    if np.any(spacing <= 0) or not np.allclose(spacing, spacing[0], rtol=2e-10, atol=0):
        raise ValueError("x must be an equally spaced increasing mesh")
    return x


def _check_times(t):
    t = np.asarray(t, dtype=float)
    if t.ndim != 1 or not np.isfinite(t).all() or np.any(t < 0):
        raise ValueError("t must be a one-dimensional array of nonnegative finite times")
    if len(t) > 1 and np.any(np.diff(t) < 0):
        raise ValueError("t must be increasing")
    return t


class TransportModel:
    def __init__(self):
        self.diffusivity = None

    @staticmethod
    def _binary_with_diffusivity(t, x, initial, diffusivity):
        """Neumann diffusion for a preparation containing one cation only."""
        t = _check_times(t)
        x = _check_mesh(x)
        initial = np.asarray(initial, dtype=float)
        if initial.shape != (len(x), 2):
            raise ValueError("initial must have shape (len(x), 2)")

        if len(t) == 0:
            return np.empty((0, len(x), 2), dtype=float)

        # DCT-I is the cosine eigenbasis on the equally spaced endpoint mesh.
        modes = np.arange(len(x))
        laplacian_eigenvalues = (
            4.0 / (x[1] - x[0]) ** 2
            * np.sin(modes * np.pi / (2.0 * (len(x) - 1))) ** 2
        )
        coefficients = dct(initial, type=1, axis=0)
        effective_scale = np.array([4.0 / 3.0, 8.0 / 3.0]) * diffusivity
        decay = np.exp(
            -t[:, None, None]
            * laplacian_eigenvalues[None, :, None]
            * effective_scale[None, None, :]
        )
        return idct(coefficients[None, :, :] * decay, type=1, axis=1)

    def fit(self, data):
        """Fit the common ionic diffusion scale from the calibration records."""
        x = _check_mesh(data["x"])
        t = _check_times(data["t"])
        initial = np.asarray(data["initial"], dtype=float)
        observed = np.asarray(data["concentration"], dtype=float)
        sigma = np.asarray(data["sigma"], dtype=float)

        if initial.ndim != 3 or initial.shape[1:] != (len(x), 2):
            raise ValueError("initial has an unexpected shape")
        expected = (len(initial), len(t), len(x), 2)
        if observed.shape != expected or sigma.shape != expected:
            raise ValueError("concentration and sigma have an unexpected shape")
        if np.any(sigma <= 0) or not np.isfinite(sigma).all():
            raise ValueError("sigma must be finite and positive")

        # Calibration preparations contain only one cation constituent.  In
        # that case the zero-current Nernst--Planck equations reduce exactly
        # to diffusion with coefficients 4D/3 (A) and 8D/3 (B).
        pure = []
        pure_indices = []
        scale = max(1.0, float(np.max(np.abs(initial))))
        for index, preparation in enumerate(initial):
            a_only = np.all(np.abs(preparation[:, 1]) <= 1e-13 * scale)
            b_only = np.all(np.abs(preparation[:, 0]) <= 1e-13 * scale)
            if a_only or b_only:
                pure.append(preparation)
                pure_indices.append(index)
        if not pure:
            raise ValueError("calibration data contain no single-cation preparation")

        def objective(log_diffusivity):
            diffusivity = np.exp(log_diffusivity)
            error = 0.0
            for preparation, index in zip(pure, pure_indices):
                prediction = self._binary_with_diffusivity(
                    t, x, preparation, diffusivity
                )
                residual = (prediction - observed[index]) / sigma[index]
                error += float(np.sum(residual * residual))
            return error

        # The log parameter makes the search positive and spans ordinary
        # aqueous transport scales comfortably.
        result = minimize_scalar(
            objective,
            bounds=(-32.0, -4.0),
            method="bounded",
            options={"xatol": 1e-11, "maxiter": 100},
        )
        if not result.success or not np.isfinite(result.fun):
            raise RuntimeError("unable to fit diffusivity")
        self.diffusivity = float(np.exp(result.x))
        return self

    def _binary(self, t, x, initial):
        if self.diffusivity is None:
            raise RuntimeError("fit the model before predicting")
        return self._binary_with_diffusivity(t, x, initial, self.diffusivity)

    def predict(self, t, x, initial):
        """Predict total A and B constituent concentrations."""
        if self.diffusivity is None or not np.isfinite(self.diffusivity) or self.diffusivity <= 0:
            raise RuntimeError("fit the model before predicting")
        t = _check_times(t)
        x = _check_mesh(x)
        initial = np.asarray(initial, dtype=float)
        if initial.shape != (len(x), 2) or not np.isfinite(initial).all():
            raise ValueError("initial must be finite and have shape (len(x), 2)")
        if len(t) == 0:
            return np.empty((0, len(x), 2), dtype=float)

        # A single-cation preparation has an exact linear solution and is
        # also the case used to calibrate D.
        if np.all(initial[:, 0] == 0.0) or np.all(initial[:, 1] == 0.0):
            return self._binary(t, x, initial)
        if t[-1] == 0.0:
            return np.repeat(initial[None, :, :], len(t), axis=0)

        count = len(x)
        dx = x[1] - x[0]
        control_volume = np.full(count, dx)
        control_volume[[0, -1]] *= 0.5
        da, db, dc = self.diffusivity * np.array([1.0, 4.0, 2.0])

        def derivative(_time, flat):
            totals = flat.reshape(count, 2)
            free_a, free_b, free_c, bound = equilibrium(totals).T
            species = np.column_stack((free_a, free_b, free_c, bound))
            face_species = 0.5 * (species[:-1] + species[1:])
            gradient = np.diff(species, axis=0) / dx

            # The electric potential gradient is selected so that
            # J_A + J_B - J_C = 0 at every face.
            denominator = (
                da * face_species[:, 0]
                + db * face_species[:, 1]
                + dc * face_species[:, 2]
            )
            numerator = -(
                da * gradient[:, 0]
                + db * gradient[:, 1]
                - dc * gradient[:, 2]
            )
            potential_gradient = np.divide(
                numerator,
                denominator,
                out=np.zeros_like(numerator),
                where=denominator > 0.0,
            )
            free_flux = -np.column_stack((da, db)) * (
                gradient[:, :2]
                + face_species[:, :2] * potential_gradient[:, None]
            )
            bound_flux = -self.diffusivity * NEUTRAL_MOBILITY * gradient[:, 3]

            constituent_flux = np.zeros((count + 1, 2), dtype=float)
            constituent_flux[1:-1] = free_flux + bound_flux[:, None]
            return (-np.diff(constituent_flux, axis=0) / control_volume[:, None]).ravel()

        # The stencil couples neighboring spatial nodes and both measured
        # channels.  This sparsity pattern helps BDF remain inexpensive on
        # the calibration mesh while retaining the stiff fast-equilibrium
        # behavior.
        sparsity = diags(
            [np.ones(2 * count - abs(offset)) for offset in range(-3, 4)],
            range(-3, 4),
            shape=(2 * count, 2 * count),
            format="csr",
        )
        result = solve_ivp(
            derivative,
            (0.0, float(t[-1])),
            initial.ravel(),
            method="BDF",
            t_eval=t,
            rtol=2e-8,
            atol=2e-10,
            jac_sparsity=sparsity,
        )
        if not result.success:
            raise RuntimeError(result.message)
        prediction = result.y.T.reshape(len(t), count, 2)
        if t[0] == 0.0:
            prediction[0] = initial
        return prediction


def fluctuation_matrices(totals, diffusivity):
    """Return local total susceptibility and zero-current mobility matrices."""
    totals = np.asarray(totals, dtype=float)
    if totals.shape != (2,):
        raise ValueError("totals must have shape (2,)")
    if not np.all(np.isfinite(totals)) or np.any(totals <= 0):
        raise ValueError("totals must be finite and positive")
    if not np.isfinite(diffusivity) or diffusivity <= 0:
        raise ValueError("diffusivity must be finite and positive")

    concentration = equilibrium(totals)
    charge = np.array([1.0, 1.0, -1.0, 0.0])
    constituents = np.array(
        [[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0]]
    )
    population = np.diag(concentration)

    # Fast charge relaxation projects local ideal number fluctuations onto
    # the electroneutral subspace.
    charge_variance = charge @ population @ charge
    conditional = population - np.outer(population @ charge, population @ charge) / charge_variance
    susceptibility = constituents @ conditional @ constituents.T

    species_mobility = np.diag(
        diffusivity * np.array([1.0, 4.0, 2.0, NEUTRAL_MOBILITY]) * concentration
    )
    mobile_charge = species_mobility @ charge
    charge_mobility = charge @ species_mobility @ charge
    constrained = species_mobility - np.outer(mobile_charge, mobile_charge) / charge_mobility
    mobility = constituents @ constrained @ constituents.T
    return susceptibility, mobility


def equilibrium_correlations(t, totals, harmonic, diffusivity):
    """Calculate the normalized two-time covariance for one cosine mode."""
    t = _check_times(t)
    totals = np.asarray(totals, dtype=float)
    harmonic_value = int(harmonic)
    if harmonic_value != harmonic or not 1 <= harmonic_value <= 8:
        raise ValueError("harmonic must be an integer from 1 through 8")
    if len(t) == 0:
        return np.empty((0, 2, 2), dtype=float)

    susceptibility, mobility = fluctuation_matrices(totals, diffusivity)
    relaxation = (
        (harmonic_value * np.pi / CELL_LENGTH) ** 2
        * mobility
        @ np.linalg.inv(susceptibility)
    )

    # The requested predictor uses independent ideal total-constituent
    # fluctuations at equal time; the chemical/electrical response controls
    # their subsequent decay.
    equal_time = np.diag(totals)
    return np.array([expm(-float(time) * relaxation) @ equal_time for time in t])


def correlate(self, t, totals, harmonic=1):
    if self.diffusivity is None:
        raise RuntimeError("fit the model before calculating correlations")
    return equilibrium_correlations(t, totals, harmonic, self.diffusivity)


TransportModel.correlate = correlate
