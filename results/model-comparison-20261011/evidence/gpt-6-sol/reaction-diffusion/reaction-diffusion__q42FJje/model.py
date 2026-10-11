"""Electroneutral transport and equilibrium fluctuations in the cell."""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize_scalar
from scipy.sparse import diags

ASSOCIATION = 0.5
NEUTRAL_MOBILITY = 0.15
CELL_LENGTH = 0.001


def _linear_cosine_coefficients(initial, modes):
    """Cosine coefficients of the piecewise linear concentration profiles."""
    cells = initial.shape[-2] - 1
    nodes = np.arange(cells + 1) / cells
    harmonics = np.arange(modes + 1)
    basis = np.cos(np.pi * np.outer(nodes, harmonics))
    weights = np.ones(cells + 1)
    weights[[0, -1]] = 0.5
    coefficients = (2 / cells) * np.einsum(
        "...xi,xk->...ki", initial * weights[:, None], basis
    ) * np.sinc(harmonics / (2 * cells))[..., None] ** 2
    coefficients[..., 0, :] *= 0.5
    return coefficients, basis


def equilibrium(total):
    """Return free A, free B, free C, and bound cluster concentrations."""
    total = np.asarray(total, dtype=float)
    a, b = total[..., 0], total[..., 1]
    bound = np.zeros_like(a)
    for _ in range(40):
        free_a, free_b = a - bound, b - bound
        anion = free_a + free_b
        residual = bound - ASSOCIATION * free_a * free_b * anion**2
        slope = 1 + ASSOCIATION * (anion**3 + 4 * free_a * free_b * anion)
        updated = np.clip(bound - residual / slope, 0, np.minimum(a, b))
        if np.max(np.abs(updated - bound)) < 1e-14:
            bound = updated
            break
        bound = updated
    return np.stack((a - bound, b - bound, a + b - 2 * bound, bound), axis=-1)


class TransportModel:
    def __init__(self):
        self.diffusivity = None

    def fit(self, data):
        """Weighted least squares fit of the one unknown ionic diffusion scale."""
        x = np.asarray(data["x"], dtype=float)
        t = np.asarray(data["t"], dtype=float)
        initial = np.asarray(data["initial"], dtype=float)
        measured = np.asarray(data["concentration"], dtype=float)
        sigma = np.asarray(data["sigma"], dtype=float)
        length = x[-1] - x[0]
        modes = 4 * (len(x) - 1)
        wave_numbers_sq = (np.arange(modes + 1) * np.pi / length) ** 2
        coefficients, basis = _linear_cosine_coefficients(initial, modes)
        ambipolar = np.array([4 / 3, 8 / 3])

        def loss(log_diffusivity):
            rates = np.exp(log_diffusivity) * wave_numbers_sq[:, None] * ambipolar
            decay = np.exp(-t[:, None, None] * rates[None, :, :])
            predicted = np.einsum(
                "rki,tki,xk->rtxi", coefficients, decay, basis, optimize=True,
            )
            predicted[:, t == 0] = initial[:, None]
            return np.mean(((predicted - measured) / sigma) ** 2)

        result = minimize_scalar(
            loss, bounds=(np.log(1e-14), np.log(1e-6)), method="bounded",
            options={"xatol": 1e-12},
        )
        if not result.success:
            raise RuntimeError(result.message)
        self.diffusivity = float(np.exp(result.x))
        return self

    def _binary(self, t, x, initial):
        length = x[-1] - x[0]
        modes = 8 * (len(x) - 1)
        wave_numbers_sq = (np.arange(modes + 1) * np.pi / length) ** 2
        coefficients, basis = _linear_cosine_coefficients(initial, modes)
        rates = self.diffusivity * wave_numbers_sq[:, None] * np.array([4 / 3, 8 / 3])
        decay = np.exp(-t[:, None, None] * rates[None, :, :])
        predicted = np.einsum(
            "ki,tki,xk->txi", coefficients, decay, basis, optimize=True,
        )
        predicted[t == 0] = initial
        return predicted

    def predict(self, t, x, initial):
        t = np.asarray(t, dtype=float)
        x = np.asarray(x, dtype=float)
        initial = np.asarray(initial, dtype=float)
        if len(t) == 0:
            return np.empty((0, len(x), 2))
        if t[-1] == 0:
            return initial[None, :, :].copy()
        if np.all(initial[:, 0] == 0) or np.all(initial[:, 1] == 0):
            return self._binary(t, x, initial)

        count = len(x)
        dx = x[1] - x[0]
        width = np.full(count, dx)
        width[[0, -1]] *= 0.5
        ion_diffusivities = self.diffusivity * np.array([1.0, 4.0, 2.0])

        def derivative(_time, flat):
            species = equilibrium(flat.reshape(count, 2))
            face = (species[:-1] + species[1:]) / 2
            gradient = np.diff(species, axis=0) / dx
            electric_gradient = -(
                ion_diffusivities[0] * gradient[:, 0]
                + ion_diffusivities[1] * gradient[:, 1]
                - ion_diffusivities[2] * gradient[:, 2]
            ) / (face[:, :3] @ ion_diffusivities)
            ion_flux = -ion_diffusivities[:2] * (
                gradient[:, :2] + face[:, :2] * electric_gradient[:, None]
            )
            cluster_flux = -self.diffusivity * NEUTRAL_MOBILITY * gradient[:, 3]
            flux = np.zeros((count + 1, 2))
            flux[1:-1] = ion_flux + cluster_flux[:, None]
            return (-np.diff(flux, axis=0) / width[:, None]).ravel()

        jac_sparsity = diags(
            [np.ones(2 * count - abs(offset)) for offset in range(-3, 4)],
            range(-3, 4), format="csr",
        )
        result = solve_ivp(
            derivative, (0, t[-1]), initial.ravel(), method="BDF", t_eval=t,
            rtol=1e-8, atol=1e-10, jac_sparsity=jac_sparsity,
        )
        if not result.success:
            raise RuntimeError(result.message)
        return result.y.T.reshape(len(t), count, 2)

    def correlate(self, t, totals, harmonic=1):
        """Stationary two-time covariance of a nonzero cosine mode."""
        t = np.asarray(t, dtype=float)
        if len(t) == 0:
            return np.empty((0, 2, 2))
        if not isinstance(harmonic, (int, np.integer)) or not 1 <= harmonic <= 8:
            raise ValueError("harmonic must be an integer from 1 to 8")
        susceptibility, mobility = fluctuation_matrices(totals, self.diffusivity)

        # Whitening makes the relaxation symmetric and preserves symmetry at every lag.
        values, vectors = np.linalg.eigh(susceptibility)
        square_root = (vectors * np.sqrt(values)) @ vectors.T
        inverse_root = (vectors / np.sqrt(values)) @ vectors.T
        relaxation = inverse_root @ mobility @ inverse_root
        rates, modes = np.linalg.eigh(relaxation)
        amplitudes = square_root @ modes
        wave_number_sq = (harmonic * np.pi / CELL_LENGTH) ** 2
        decay = np.exp(-t[:, None] * wave_number_sq * rates[None, :])
        return np.einsum("ik,tk,jk->tij", amplitudes, decay, amplitudes)


def fluctuation_matrices(totals, diffusivity):
    """Static susceptibility and charge-constrained constituent mobility."""
    concentrations = equilibrium(np.asarray(totals, dtype=float))
    charge = np.array([1.0, 1.0, -1.0, 0.0])
    constituents = np.array([[1.0, 0.0, 0.0, 1.0],
                             [0.0, 1.0, 0.0, 1.0]])

    # Ideal species number fluctuations, conditioned on local neutrality.
    population = np.diag(concentrations)
    charged_population = population @ charge
    conditional = population - np.outer(charged_population, charged_population) / (
        charge @ charged_population
    )
    susceptibility = constituents @ conditional @ constituents.T

    # The fluctuating electric field removes the instantaneous charge flux.
    population_mobility = np.diag(
        diffusivity * np.array([1.0, 4.0, 2.0, NEUTRAL_MOBILITY])
        * concentrations
    )
    charged_mobility = population_mobility @ charge
    constrained = population_mobility - np.outer(charged_mobility, charged_mobility) / (
        charge @ charged_mobility
    )
    mobility = constituents @ constrained @ constituents.T
    return susceptibility, mobility


def equilibrium_correlations(t, totals, harmonic, diffusivity):
    """Standalone version of the model's correlation predictor."""
    model = TransportModel()
    model.diffusivity = diffusivity
    return model.correlate(t, totals, harmonic)
