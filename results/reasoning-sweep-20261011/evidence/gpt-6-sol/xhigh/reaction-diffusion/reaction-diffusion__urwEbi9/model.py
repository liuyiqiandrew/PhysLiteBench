"""Electroneutral transport and equilibrium fluctuations in the closed cell."""

import numpy as np
from scipy.fft import dct, idct
from scipy.integrate import solve_ivp
from scipy.optimize import minimize_scalar
from scipy.sparse import diags


ASSOCIATION = 0.5
NEUTRAL_MOBILITY = 0.15
CELL_LENGTH = 0.001
ION_DIFFUSIVITIES = np.array([1.0, 4.0, 2.0, NEUTRAL_MOBILITY])
CHARGE = np.array([1.0, 1.0, -1.0, 0.0])
# Rows map free A, free B, free C, and P to the two measured totals.
CONSTITUENTS = np.array([[1.0, 0.0, 0.0, 1.0],
                         [0.0, 1.0, 0.0, 1.0]])


def equilibrium(total):
    """Return local (A, B, C, P) concentrations from (total A, total B)."""
    a, b = np.moveaxis(np.asarray(total, dtype=float), -1, 0)
    bound = np.zeros_like(a)
    for _ in range(40):
        free_a = a - bound
        free_b = b - bound
        anion = free_a + free_b
        residual = bound - ASSOCIATION * free_a * free_b * anion**2
        slope = 1 + ASSOCIATION * (anion**3 + 4 * free_a * free_b * anion)
        change = residual / slope
        updated = np.clip(bound - change, 0.0, np.minimum(a, b))
        if np.max(np.abs(updated - bound)) < 1e-14:
            bound = updated
            break
        bound = updated
    return np.stack((a - bound, b - bound, a + b - 2 * bound, bound), axis=-1)


def _binary_profiles(t, x, initial, diffusivity):
    """Exact solution of the discrete no-flux binary diffusion equations."""
    n = len(x) - 1
    eigenvalue = 4 / (x[1] - x[0])**2 * np.sin(
        np.arange(n + 1) * np.pi / (2 * n))**2
    coefficient = dct(initial, type=1, axis=-2)
    rates = diffusivity * eigenvalue[:, None] * np.array([4 / 3, 8 / 3])
    return idct(np.exp(-t[:, None, None] * rates[None, :, :])
                * coefficient[None, :, :], type=1, axis=1)


class TransportModel:
    def __init__(self):
        self.diffusivity = None

    def fit(self, data):
        """Weighted maximum likelihood fit of the common diffusion scale."""
        x = np.asarray(data["x"], dtype=float)
        t = np.asarray(data["t"], dtype=float)
        initial = np.asarray(data["initial"], dtype=float)
        observed = np.asarray(data["concentration"], dtype=float)
        sigma = np.asarray(data["sigma"], dtype=float)

        n = len(x) - 1
        eigenvalue = 4 / (x[1] - x[0])**2 * np.sin(
            np.arange(n + 1) * np.pi / (2 * n))**2
        coefficients = dct(initial, type=1, axis=1)
        rates = eigenvalue[None, None, :, None] * np.array([4 / 3, 8 / 3])

        def error(log_diffusivity):
            decay = np.exp(-np.exp(log_diffusivity)
                           * t[None, :, None, None] * rates)
            predicted = idct(coefficients[:, None, :, :] * decay,
                             type=1, axis=2)
            return np.sum(((predicted - observed) / sigma)**2)

        # The calibration preparations each contain only one cation. Their
        # cluster concentration is zero, and the two ambipolar rates are
        # respectively 4D/3 and 8D/3.
        time_scale = (x[-1] - x[0])**2 / max(float(t[-1]), 1e-12)
        bounds = (np.log(time_scale) - 12, np.log(time_scale) + 12)
        optimum = minimize_scalar(error, bounds=bounds, method="bounded",
                                  options={"xatol": 1e-11})
        if not optimum.success:
            raise RuntimeError("Diffusivity fit failed")
        self.diffusivity = float(np.exp(optimum.x))
        return self

    def predict(self, t, x, initial):
        t = np.asarray(t, dtype=float)
        x = np.asarray(x, dtype=float)
        initial = np.asarray(initial, dtype=float)
        if t[-1] == 0:
            return initial[None, :, :].copy()
        if np.all(initial[:, 0] == 0) or np.all(initial[:, 1] == 0):
            return _binary_profiles(t, x, initial, self.diffusivity)

        count = len(x)
        dx = x[1] - x[0]
        width = np.full(count, dx)
        width[[0, -1]] *= 0.5
        da, db, dc = self.diffusivity * ION_DIFFUSIVITIES[:3]

        def derivative(_time, flat):
            species = equilibrium(flat.reshape(count, 2))
            face = (species[:-1] + species[1:]) / 2
            gradient = np.diff(species, axis=0) / dx
            charge_gradient = da * gradient[:, 0] + db * gradient[:, 1] - dc * gradient[:, 2]
            conductivity = (da * face[:, 0] + db * face[:, 1]
                            + dc * face[:, 2])
            # In a wholly empty face both the charge flux and conductivity
            # vanish; its field can be set to zero without changing any flux.
            field = np.zeros_like(conductivity)
            np.divide(-charge_gradient, conductivity, out=field,
                      where=conductivity > 0)
            ionic_flux = -np.array([da, db]) * (
                gradient[:, :2] + face[:, :2] * field[:, None])
            bound_flux = -self.diffusivity * NEUTRAL_MOBILITY * gradient[:, 3]
            flux = np.zeros((count + 1, 2))
            flux[1:-1] = ionic_flux + bound_flux[:, None]
            return (-np.diff(flux, axis=0) / width[:, None]).ravel()

        sparsity = diags([np.ones(2 * count - abs(i)) for i in range(-3, 4)],
                         range(-3, 4), format="csr")
        result = solve_ivp(derivative, (0, t[-1]), initial.ravel(),
                           method="BDF", t_eval=t, rtol=1e-8, atol=1e-10,
                           jac_sparsity=sparsity)
        if not result.success:
            raise RuntimeError(result.message)
        return result.y.T.reshape(len(t), count, 2)

    def correlate(self, t, totals, harmonic=1):
        """Two-time covariance of the specified nonzero cosine mode."""
        if not isinstance(harmonic, (int, np.integer)) or not 1 <= harmonic <= 8:
            raise ValueError("harmonic must be an integer from 1 to 8")
        t = np.asarray(t, dtype=float)
        covariance, mobility = fluctuation_matrices(totals, self.diffusivity)

        # Symmetric coordinates preserve reciprocity and positivity at every
        # lag: S(t) = C^(1/2) exp[-q² C^(-1/2) M C^(-1/2) t] C^(1/2).
        values, vectors = np.linalg.eigh(covariance)
        root = (vectors * np.sqrt(values)) @ vectors.T
        inverse_root = (vectors / np.sqrt(values)) @ vectors.T
        generator = inverse_root @ mobility @ inverse_root
        rates, modes = np.linalg.eigh((generator + generator.T) / 2)
        basis = root @ modes
        wave_number = harmonic * np.pi / CELL_LENGTH
        decay = np.exp(-t[:, None] * wave_number**2 * rates[None, :])
        return np.einsum("ik,tk,jk->tij", basis, decay, basis)


def fluctuation_matrices(totals, diffusivity):
    """Static susceptibility and charge-projected transport mobility."""
    species = equilibrium(np.asarray(totals, dtype=float))
    population = np.diag(species)
    charge_population = population @ CHARGE
    # Charge neutrality conditions the ideal species-number covariance.
    conditioned = population - np.outer(charge_population, charge_population) / (
        CHARGE @ charge_population)
    covariance = CONSTITUENTS @ conditioned @ CONSTITUENTS.T

    species_mobility = np.diag(diffusivity * ION_DIFFUSIVITIES * species)
    charge_mobility = species_mobility @ CHARGE
    # The fluctuating electric field removes instantaneous charge flux too.
    projected = species_mobility - np.outer(charge_mobility, charge_mobility) / (
        CHARGE @ charge_mobility)
    mobility = CONSTITUENTS @ projected @ CONSTITUENTS.T
    return covariance, mobility
