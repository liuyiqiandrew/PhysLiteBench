"""Electroneutral transport with locally equilibrated neutral association."""

import numpy as np
from scipy.fft import dct, idct
from scipy.integrate import solve_ivp
from scipy.linalg import eigh
from scipy.optimize import minimize_scalar
from scipy.sparse import diags
from scipy.special import erf


ASSOCIATION = 0.5
NEUTRAL_MOBILITY = 0.15
CELL_LENGTH = 0.001
SPECIES_MOBILITIES = np.array([1.0, 4.0, 2.0, NEUTRAL_MOBILITY])
CHARGE = np.array([1.0, 1.0, -1.0, 0.0])
CONSTITUENTS = np.array([[1.0, 0.0, 0.0, 1.0],
                         [0.0, 1.0, 0.0, 1.0]])


def equilibrium(total):
    """Return free A, free B, free C, and P for totals of shape (..., 2)."""
    # Clipping also accommodates roundoff at the boundary of the positive
    # concentration domain during implicit integration.
    total = np.maximum(np.asarray(total, dtype=float), 0.0)
    a, b = total[..., 0], total[..., 1]
    bound = np.zeros_like(a)
    for _ in range(64):
        free_a, free_b = a - bound, b - bound
        anion = free_a + free_b
        residual = bound - ASSOCIATION * free_a * free_b * anion**2
        slope = 1.0 + ASSOCIATION * (anion**3 + 4 * free_a * free_b * anion)
        updated = np.clip(bound - residual / slope, 0.0, np.minimum(a, b))
        converged = np.all(np.abs(updated - bound) <= 2e-14 * (1.0 + bound))
        bound = updated
        if converged:
            break
    return np.stack((a - bound, b - bound, a + b - 2 * bound, bound), axis=-1)


def _times(t):
    t = np.asarray(t, dtype=float)
    if (t.ndim != 1 or not np.isfinite(t).all() or np.any(t < 0)
            or np.any(np.diff(t) <= 0)):
        raise ValueError("Times must be finite, nonnegative, and strictly increasing.")
    return t


def _binary_factors(t, x, diffusivity):
    """Neumann heat multipliers for piecewise linear initial concentrations.

    The cosine transform diagonalizes sampled heat evolution. Linear
    interpolation introduces sinc-squared weights on every Fourier alias.
    For very short times, an equivalent heat-kernel integral over the linear
    hat functions avoids summing a large number of aliases.
    """
    n = len(x) - 1
    theta = np.arange(n + 1) * np.pi / n
    tau = (t[:, None] * diffusivity * np.array([4 / 3, 8 / 3])
           / (x[1] - x[0])**2)
    flat_tau = tau.ravel()
    factors = np.ones((flat_tau.size, n + 1))

    long = flat_tau >= 0.1
    wave = theta[:, None] + 2 * np.pi * np.arange(-4, 5)[None, :]
    factors[long] = np.sum(
        np.sinc(wave[None, :, :] / (2 * np.pi))**2
        * np.exp(-flat_tau[long, None, None] * wave[None, :, :]**2), axis=2)

    short = (flat_tau > 0) & ~long
    if np.any(short):
        s = np.arange(-1, 7, dtype=float)[None, :]
        tau_short = flat_tau[short, None]
        root = np.sqrt(tau_short)
        primitive = (0.5 * s * erf(s / (2 * root))
                     + root / np.sqrt(np.pi) * np.exp(-s**2 / (4 * tau_short)))
        weights = np.diff(primitive, n=2, axis=1)
        weights[:, 1:] *= 2
        factors[short] = weights @ np.cos(np.arange(6)[:, None] * theta)

    # The constant mode is exactly conserved, including at roundoff level.
    factors[:, 0] = 1.0
    return factors.reshape(len(t), 2, n + 1).transpose(0, 2, 1)


def fluctuation_matrices(totals, diffusivity):
    """Return the equilibrium susceptibility and Onsager mobility of totals.

    Chemical equilibrium leaves constituent amounts as slow coordinates.
    Projecting the ideal species susceptibility onto those coordinates
    automatically includes equilibrium association fluctuations. Charge is
    constrained locally, so its Schur complement is taken first. The same
    zero-current projection must be applied to Brownian and mean fluxes.
    """
    concentration = equilibrium(totals)
    population = np.diag(concentration)
    population_charge = population @ CHARGE
    conditional = (population - np.outer(population_charge, population_charge)
                   / (CHARGE @ population_charge))
    susceptibility = CONSTITUENTS @ conditional @ CONSTITUENTS.T

    mobile = np.diag(diffusivity * SPECIES_MOBILITIES * concentration)
    mobile_charge = mobile @ CHARGE
    constrained = (mobile - np.outer(mobile_charge, mobile_charge)
                   / (CHARGE @ mobile_charge))
    mobility = CONSTITUENTS @ constrained @ CONSTITUENTS.T
    return susceptibility, mobility


def equilibrium_correlations(t, totals, harmonic, diffusivity):
    covariance, mobility = fluctuation_matrices(totals, diffusivity)
    # M v = lambda C v, with v.T C v = I. This symmetric formulation of
    # exp(-q^2 M C^-1 t) C preserves detailed balance and nonnegative decay.
    rates, modes = eigh(mobility, covariance)
    amplitudes = covariance @ modes
    decay = np.exp(-np.asarray(t)[:, None]
                   * (harmonic * np.pi / CELL_LENGTH)**2 * rates[None, :])
    result = np.einsum("ai,ti,bi->tab", amplitudes, decay, amplitudes)
    result[np.asarray(t) == 0] = covariance
    return result


class TransportModel:
    def __init__(self):
        self.diffusivity = None

    def _require_fitted(self):
        if (self.diffusivity is None or not np.isfinite(self.diffusivity)
                or self.diffusivity <= 0):
            raise RuntimeError("Fit the model before requesting predictions.")

    def fit(self, data):
        """Fit the common positive diffusion scale by Gaussian likelihood."""
        t = _times(data["t"])
        x = np.asarray(data["x"], dtype=float)
        initial = np.asarray(data["initial"], dtype=float)
        observed = np.asarray(data["concentration"], dtype=float)
        sigma = np.asarray(data["sigma"], dtype=float)
        if not np.any(t > 0):
            raise ValueError("Calibration needs measurements at positive times.")
        if np.any(sigma <= 0) or not np.isfinite(sigma).all():
            raise ValueError("Calibration uncertainties must be finite and positive.")
        if not np.all(np.any(np.all(initial == 0, axis=1), axis=1)):
            raise ValueError("Calibration preparations must contain only one cation constituent.")

        coefficients = dct(initial, type=1, axis=1)

        def objective(log_scale):
            diffusion = reference * np.exp(log_scale)
            factors = _binary_factors(t, x, diffusion)
            predicted = idct(coefficients[:, None, :, :] * factors[None, :, :, :],
                             type=1, axis=2)
            residual = (predicted - observed) / sigma
            return np.sum(residual**2)

        reference = ((x[-1] - x[0]) / np.pi)**2 / np.median(t[t > 0])
        # A coarse search prevents a very fast, fully relaxed plateau from
        # concealing the identifiable minimum in the scalar optimization.
        grid = np.linspace(-12.0, 12.0, 25)
        best = int(np.argmin([objective(value) for value in grid]))
        lower, upper = grid[max(0, best - 1)], grid[min(len(grid) - 1, best + 1)]
        result = minimize_scalar(objective, bounds=(lower, upper), method="bounded",
                                 options={"xatol": 1e-10})
        if not result.success or not np.isfinite(result.fun):
            raise RuntimeError("Diffusivity calibration failed.")
        self.diffusivity = float(reference * np.exp(result.x))
        return self

    def _binary(self, t, x, initial):
        coefficients = dct(initial, type=1, axis=0)
        factors = _binary_factors(t, x, self.diffusivity)
        result = idct(factors * coefficients[None, :, :], type=1, axis=1)
        result[t == 0] = initial
        return result

    def predict(self, t, x, initial):
        """Predict total A and B concentrations in the closed cell."""
        self._require_fitted()
        t = _times(t)
        x, initial = np.asarray(x, dtype=float), np.asarray(initial, dtype=float)
        if (x.ndim != 1 or len(x) < 2 or not np.isfinite(x).all()
                or np.any(np.diff(x) <= 0)
                or not np.allclose(np.diff(x), x[1] - x[0], rtol=1e-8, atol=1e-15)
                or not np.isclose(x[0], 0, atol=1e-15)
                or not np.isclose(x[-1], CELL_LENGTH, rtol=1e-8, atol=1e-15)):
            raise ValueError("x must be an equally spaced mesh spanning the cell.")
        if (initial.shape != (len(x), 2) or not np.isfinite(initial).all()
                or np.any(initial < 0)):
            raise ValueError("Initial totals must be finite, nonnegative, and have shape (len(x), 2).")
        if len(t) == 0:
            return np.empty((0, len(x), 2))
        if t[-1] == 0 or np.all(initial == initial[0]):
            return np.broadcast_to(initial, (len(t), *initial.shape)).copy()
        if np.all(initial[:, 0] == 0) or np.all(initial[:, 1] == 0):
            return self._binary(t, x, initial)

        # Retain every supplied node, and interpolate the preparation linearly
        # onto a finer finite-volume mesh to resolve mixed-species transport.
        refinement = max(1, int(np.ceil(256 / (len(x) - 1))))
        mesh = np.linspace(x[0], x[-1], refinement * (len(x) - 1) + 1)
        prepared = np.column_stack([np.interp(mesh, x, initial[:, i]) for i in range(2)])
        count, dx = len(mesh), mesh[1] - mesh[0]
        width = np.full(count, dx)
        width[[0, -1]] *= 0.5
        da, db, dc, dp = self.diffusivity * SPECIES_MOBILITIES

        def derivative(time, flat):
            species = equilibrium(flat.reshape(count, 2))
            face = (species[:-1] + species[1:]) / 2
            gradient = np.diff(species, axis=0) / dx
            conductivity = da * face[:, 0] + db * face[:, 1] + dc * face[:, 2]
            charge_gradient = da * gradient[:, 0] + db * gradient[:, 1] - dc * gradient[:, 2]
            field = np.divide(-charge_gradient, conductivity,
                              out=np.zeros_like(conductivity), where=conductivity > 0)
            free_flux = -np.array([da, db]) * (gradient[:, :2] + face[:, :2] * field[:, None])
            bound_flux = -dp * gradient[:, 3]
            flux = np.zeros((count + 1, 2))
            flux[1:-1] = free_flux + bound_flux[:, None]
            return (-np.diff(flux, axis=0) / width[:, None]).ravel()

        sparsity = diags([np.ones(2 * count - abs(i)) for i in range(-3, 4)],
                         range(-3, 4), format="csr")
        result = solve_ivp(derivative, (0, t[-1]), prepared.ravel(), method="BDF",
                           t_eval=t, rtol=2e-8, atol=2e-10, jac_sparsity=sparsity)
        if not result.success:
            raise RuntimeError(result.message)
        prediction = result.y.T.reshape(len(t), count, 2)[:, ::refinement, :].copy()
        prediction[t == 0] = initial
        return prediction

    def correlate(self, t, totals, harmonic=1):
        """Equilibrium two-time covariance of a nonzero cosine mode."""
        self._require_fitted()
        t, totals = _times(t), np.asarray(totals, dtype=float)
        if totals.shape != (2,) or not np.isfinite(totals).all() or np.any(totals <= 0):
            raise ValueError("totals must contain two finite positive concentrations.")
        if not np.isscalar(harmonic) or harmonic != int(harmonic) or not 1 <= harmonic <= 8:
            raise ValueError("harmonic must be an integer from 1 through 8.")
        return equilibrium_correlations(t, totals, int(harmonic), self.diffusivity)
