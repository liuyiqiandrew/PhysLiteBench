"""Electroneutral, locally equilibrated constituent transport in a closed cell."""

import numpy as np
from scipy.fft import dct, idct
from scipy.integrate import solve_ivp
from scipy.optimize import minimize_scalar
from scipy.sparse import diags

ASSOCIATION = 0.5
NEUTRAL_MOBILITY = 0.15
CELL_LENGTH = 0.001


def equilibrium(total):
    """Return free A, B, C and bound P at the specified constituent totals."""
    total = np.asarray(total, dtype=float)
    a, b = total[..., 0], total[..., 1]
    bound = np.zeros_like(a)
    # The mass-action residual is strictly increasing on [0, min(a,b)].
    for _ in range(100):
        free_a, free_b = a - bound, b - bound
        anion = free_a + free_b
        residual = bound - ASSOCIATION * free_a * free_b * anion**2
        slope = 1 + ASSOCIATION * (anion**3 + 4 * free_a * free_b * anion)
        updated = np.clip(bound - residual / slope, 0, np.minimum(a, b))
        change = np.max(np.abs(updated - bound))
        bound = updated
        if change < 1e-14:
            break
    return np.stack((a - bound, b - bound, a + b - 2 * bound, bound), axis=-1)


def fluctuation_matrices(totals, diffusivity):
    """Thermodynamic susceptibility and zero-current Onsager mobility.

    Ideal species number fluctuations are conditioned on zero local charge.
    Mapping these to conserved constituents includes the equilibrium reaction
    fluctuations without introducing an independent noise in those totals.
    The same charge projection of Brownian fluxes uses mobility, not population.
    """
    concentration = equilibrium(totals)
    charge = np.array([1., 1., -1., 0.])
    constituents = np.array([[1., 0., 0., 1.], [0., 1., 0., 1.]])

    def projected(weights):
        charged = weights * charge
        matrix = np.diag(weights) - np.outer(charged, charged) / (charge @ charged)
        return constituents @ matrix @ constituents.T

    susceptibility = projected(concentration)
    mobility = projected(diffusivity * np.array([1., 4., 2., NEUTRAL_MOBILITY]) * concentration)
    return susceptibility, mobility


def equilibrium_correlations(t, totals, harmonic, diffusivity):
    covariance, mobility = fluctuation_matrices(totals, diffusivity)
    # Whiten by the static susceptibility: the resulting generator is symmetric
    # positive definite, preserving detailed balance at every time lag.
    values, vectors = np.linalg.eigh(covariance)
    root = (vectors * np.sqrt(values)) @ vectors.T
    inverse_root = (vectors / np.sqrt(values)) @ vectors.T
    generator = inverse_root @ mobility @ inverse_root
    rates, modes = np.linalg.eigh((generator + generator.T) / 2)
    amplitudes = root @ modes
    decay = np.exp(-np.asarray(t)[:, None] * (harmonic * np.pi / CELL_LENGTH)**2 * rates)
    return np.einsum('ik,tk,jk->tij', amplitudes, decay, amplitudes)


class TransportModel:
    def __init__(self):
        self.diffusivity = None

    def fit(self, data):
        """Gaussian maximum-likelihood fit of the single diffusion scale."""
        times, x = np.asarray(data['t']), np.asarray(data['x'])
        initial = np.asarray(data['initial'])
        observed, sigma = np.asarray(data['concentration']), np.asarray(data['sigma'])

        def objective(log_diffusivity):
            self.diffusivity = float(np.exp(log_diffusivity))
            predicted = np.array([self.predict(times, x, profile) for profile in initial])
            return np.sum(((predicted - observed) / sigma)**2)

        result = minimize_scalar(objective, bounds=(np.log(1e-14), np.log(1e-6)),
                                 method='bounded', options={'xatol': 1e-10})
        if not result.success:
            raise RuntimeError(result.message)
        self.diffusivity = float(np.exp(result.x))
        return self

    def _binary(self, t, x, initial):
        # Exact diagonalization of the conservative, reflecting heat operator.
        count = len(x) - 1
        spatial_rate = 4 / (x[1] - x[0])**2 * np.sin(np.arange(count + 1) * np.pi / (2 * count))**2
        coefficients = dct(initial, type=1, axis=0)
        rates = spatial_rate[:, None] * self.diffusivity * np.array([4/3, 8/3])[None, :]
        return idct(np.exp(-t[:, None, None] * rates[None, :, :]) * coefficients[None, :, :],
                    type=1, axis=1)

    def predict(self, t, x, initial):
        t, x, initial = np.asarray(t, dtype=float), np.asarray(x, dtype=float), np.asarray(initial, dtype=float)
        if len(t) == 0:
            return np.empty((0, len(x), 2))
        if t[-1] == 0:
            return initial[None, :, :].copy()
        # Resolve the documented piecewise-linear preparation on a finer mesh.
        # Original sample nodes remain nodes, with half cells at reflecting ends.
        refinement = max(1, int(np.ceil(256 / (len(x) - 1))))
        mesh = np.linspace(x[0], x[-1], refinement * (len(x) - 1) + 1)
        prepared = np.column_stack([np.interp(mesh, x, initial[:, i]) for i in range(2)])
        if np.all(initial[:, 0] == 0) or np.all(initial[:, 1] == 0):
            return self._binary(t, mesh, prepared)[:, ::refinement, :]
        count, dx = len(mesh), mesh[1] - mesh[0]
        width = np.full(count, dx)
        width[[0, -1]] *= .5
        da, db, dc = self.diffusivity * np.array([1., 4., 2.])

        def derivative(time, flat):
            c = equilibrium(flat.reshape(count, 2))
            face = (c[:-1] + c[1:]) / 2
            gradient = np.diff(c, axis=0) / dx
            conductivity = da * face[:, 0] + db * face[:, 1] + dc * face[:, 2]
            numerator = da * gradient[:, 0] + db * gradient[:, 1] - dc * gradient[:, 2]
            field = np.divide(-numerator, conductivity, out=np.zeros_like(numerator), where=conductivity > 0)
            free_flux = -np.array([da, db]) * (gradient[:, :2] + face[:, :2] * field[:, None])
            bound_flux = -self.diffusivity * NEUTRAL_MOBILITY * gradient[:, 3]
            flux = np.zeros((count + 1, 2))
            flux[1:-1] = free_flux + bound_flux[:, None]
            return (-np.diff(flux, axis=0) / width[:, None]).ravel()

        sparsity = diags([np.ones(2 * count - abs(i)) for i in range(-3, 4)],
                         range(-3, 4), format='csr')
        result = solve_ivp(derivative, (0, t[-1]), prepared.ravel(), method='BDF',
                           t_eval=t, rtol=1e-8, atol=1e-10, jac_sparsity=sparsity)
        if not result.success:
            raise RuntimeError(result.message)
        return result.y.T.reshape(len(t), count, 2)[:, ::refinement, :]

    def correlate(self, t, totals, harmonic=1):
        return equilibrium_correlations(np.asarray(t, dtype=float), np.asarray(totals, dtype=float),
                                        harmonic, self.diffusivity)
